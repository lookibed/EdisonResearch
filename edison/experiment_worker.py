"""Trusted bootstrap inside a bubblewrap sandbox; stdout is an untrusted RPC stream."""
import ctypes
import errno
import json
import platform
import resource
import sys
import traceback


def confine():
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_AS, (384 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if platform.machine() != 'x86_64':
        raise RuntimeError('Seccomp profile currently requires x86_64')
    class Filter(ctypes.Structure):
        _fields_ = [('code', ctypes.c_ushort), ('jt', ctypes.c_ubyte),
                    ('jf', ctypes.c_ubyte), ('k', ctypes.c_uint)]
    class Program(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ushort), ('filter', ctypes.POINTER(Filter))]
    # Verify architecture, reject x32 calls, then deny process creation/exec/ptrace.
    instructions = [(0x20, 0, 0, 4), (0x15, 1, 0, 0xC000003E),
                    (0x06, 0, 0, 0x80000000), (0x20, 0, 0, 0),
                    (0x35, 0, 1, 0x40000000), (0x06, 0, 0, 0x00050000 | errno.EPERM)]
    for syscall in (56, 57, 58, 59, 101, 322, 435):
        instructions.extend([(0x15, 0, 1, syscall), (0x06, 0, 0, 0x00050000 | errno.EPERM)])
    instructions.append((0x06, 0, 0, 0x7FFF0000))
    filters = (Filter * len(instructions))(*(Filter(*i) for i in instructions))
    program = Program(len(instructions), filters)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) or libc.prctl(22, 2, ctypes.byref(program), 0, 0):
        raise RuntimeError('Cannot enable seccomp')


def send(value):
    sys.stdout.write(json.dumps(value, allow_nan=False) + '\n')
    sys.stdout.flush()


def main():
    request = json.loads(sys.stdin.readline())
    try:
        confine()
    except Exception:
        send({'op': 'bootstrap_error'})
        return
    send({'op': 'ready'})
    count = 0
    def call(tool, arguments):
        nonlocal count
        count += 1
        send({'op': 'call', 'tool': tool, 'arguments': arguments})
        response = json.loads(sys.stdin.readline())
        if response['status'] == 'budget_exhausted':
            raise RuntimeError('Tool call budget exhausted')
        return response
    def emit(value):
        send({'op': 'emit', 'value': value})
    try:
        code = compile(request['program'], '<experiment>', 'exec')
        exec(code, {'__name__': '__experiment__', 'call': call, 'emit': emit})
        send({'op': 'done'})
    except BaseException as exc:
        send({'op': 'experiment_error', 'error': type(exc).__name__,
              'message': str(exc)[:300], 'traceback': traceback.format_exc()[-1500:]})


if __name__ == '__main__':
    main()
