"""OS-isolated experiments. Missing bubblewrap/namespaces fail closed; no exec fallback."""
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import tempfile
import time


class SandboxUnavailable(RuntimeError):
    pass


class ExperimentExecutor:
    def __init__(self, timeout=12, max_calls=24, max_messages=64):
        if min(timeout, max_calls, max_messages) <= 0:
            raise ValueError('Positive execution limits required')
        self.timeout, self.max_calls, self.max_messages = timeout, max_calls, max_messages
        self._checked = False

    def check(self):
        if not self._checked:
            result = self.run('emit({"isolation_ready": True})', [], lambda *_: None)
            if result['status'] != 'ok':
                raise SandboxUnavailable('OS isolation preflight failed')
            self._checked = True

    def run(self, program, tools, invoke):
        if not isinstance(program, str) or not 1 <= len(program) <= 16000:
            raise ValueError('Expected program of 1..16000 characters')
        bwrap = shutil.which('bwrap')
        if not bwrap or not Path('/usr/bin/python3').exists():
            raise SandboxUnavailable('bubblewrap and /usr/bin/python3 are required')
        allowed = {t['name'] for t in tools}
        command = [bwrap, '--unshare-all', '--die-with-parent', '--new-session', '--cap-drop', 'ALL',
                   '--clearenv', '--setenv', 'PATH', '/usr/bin', '--setenv', 'LANG', 'C.UTF-8']
        for directory in ('/usr', '/lib', '/lib64'):
            if Path(directory).exists():
                command += ['--ro-bind', directory, directory]
        worker = str(Path(__file__).with_name('experiment_worker.py').resolve())
        command += ['--ro-bind', worker, '/runner.py', '--proc', '/proc', '--dev', '/dev',
                    '--tmpfs', '/tmp', '--chdir', '/tmp', '/usr/bin/python3', '-I', '-u', '/runner.py']
        evidence, emissions = [], []
        result = {'status': 'protocol_error', 'evidence': evidence, 'emissions': emissions,
                  'isolation': 'bubblewrap+seccomp', 'tool_calls': 0}
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=errors, env={'PATH': '/usr/bin:/bin'},
                                       start_new_session=True, bufsize=0)
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            deadline = time.monotonic() + self.timeout
            buffer = b''
            messages = 0
            bootstrapped = False
            def reply(value):
                process.stdin.write((json.dumps(value, allow_nan=False) + '\n').encode())
            try:
                reply({'program': program})
                while time.monotonic() < deadline:
                    ready = selector.select(min(.1, max(0, deadline - time.monotonic())))
                    if not ready:
                        continue
                    chunk = os.read(process.stdout.fileno(), 8192)
                    if not chunk:
                        if not bootstrapped:
                            result['status'] = 'sandbox_unavailable'
                        else:
                            result['status'] = 'worker_stopped'
                        break
                    buffer += chunk
                    if len(buffer) > 262144:
                        result['status'] = 'output_limit'
                        break
                    stop = False
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        messages += 1
                        if messages > self.max_messages:
                            result['status'] = 'message_limit'
                            stop = True
                            break
                        try:
                            message = json.loads(line)
                        except (ValueError, UnicodeDecodeError):
                            result['status'] = 'protocol_error'
                            stop = True
                            break
                        if type(message) is not dict:
                            result['status'] = 'protocol_error'
                            stop = True
                            break
                        op = message.get('op')
                        if op == 'ready':
                            bootstrapped = True
                        elif op == 'call':
                            if result['tool_calls'] >= self.max_calls:
                                reply({'status': 'budget_exhausted'})
                                result['status'] = 'tool_budget_exhausted'
                                stop = True
                                break
                            result['tool_calls'] += 1
                            if message.get('tool') not in allowed or type(message.get('arguments')) is not dict:
                                reply({'status': 'invalid_tool_request'})
                                continue
                            try:
                                observation = invoke(message['tool'], message['arguments'])
                            except (ValueError, RuntimeError, TypeError, KeyError):
                                reply({'status': 'tool_error'})
                                continue
                            # Only actual trusted invocations populate the authoritative ledger.
                            evidence.append(observation)
                            reply(observation)
                        elif op == 'emit':
                            emissions.append(message.get('value'))
                        elif op in ('done', 'experiment_error', 'bootstrap_error'):
                            result['status'] = {'done': 'ok', 'experiment_error': 'experiment_error',
                                                'bootstrap_error': 'sandbox_unavailable'}[op]
                            if op == 'experiment_error':
                                result['error'] = message.get('error')
                                result['diagnostic'] = message.get('message')
                            stop = True
                            break
                        else:
                            result['status'] = 'protocol_error'
                            stop = True
                            break
                    if stop:
                        break
                else:
                    result['status'] = 'timeout'
            except (BrokenPipeError, OSError):
                result['status'] = 'worker_stopped'
            finally:
                selector.close()
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2)
                process.stdin.close()
                process.stdout.close()
        if result['status'] == 'sandbox_unavailable':
            raise SandboxUnavailable('OS isolation unavailable; experiment was not run without isolation')
        return result
