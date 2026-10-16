"""Trusted evaluator process. This is process separation, not an OS sandbox."""
from dataclasses import asdict
import multiprocessing
from .ideas import Construction, Primitives
from .wire import construction_to_json


def _serve(connection, profile, seed, budget, probe_messages, holdout_messages):
    # Only this child constructs the world and owns the experiment memory.
    from .composition import CompositionWorld, ExperimentMemory
    from .wire import construction_from_json
    try:
        world = CompositionWorld(profile, seed)
        memory = ExperimentMemory(world, budget, probe_messages)
        connection.send({"primitives": asdict(world.primitives),
                         "max_cost": memory.max_cost, "target_ber": memory.target_ber})
        while True:
            request = connection.recv()
            if request.get("op") == "close":
                return
            try:
                if request.get("op") == "probe":
                    c = construction_from_json(request["construction"], world.primitives)
                    connection.send({"outcome": asdict(memory.probe(c))})
                elif request.get("op") == "finish":
                    try:
                        selected = memory.best()
                    except ValueError:
                        selected = None
                    held = memory.finalize(selected, holdout_messages) if selected else None
                    m = held.measurement if held else None
                    connection.send({
                        "profile": profile, "seed": seed,
                        "selected": asdict(selected) if selected else None,
                        "holdout": asdict(held) if held else None,
                        "success": bool(m and m.cost <= memory.max_cost
                                        and m.ber <= memory.target_ber),
                        "observations": [{"construction": asdict(c), "outcome": asdict(o)}
                                         for c, o in memory.observations],
                        "duplicate_requests": memory.duplicate_requests,
                    })
                    return  # No second holdout and no probes after selection.
                else:
                    connection.send({"error": "unsupported_operation"})
            except (KeyError, ValueError, RuntimeError, TypeError):
                connection.send({"error": "invalid_request"})
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        # Do not return private world details in an exception or traceback.
        try:
            connection.send({"error": "evaluator_failed"})
        except (EOFError, BrokenPipeError):
            pass
    finally:
        connection.close()


class EvaluatorProcess:
    def __init__(self, profile: str, seed: int, budget: int,
                 probe_messages: int = 512, holdout_messages: int = 8192, timeout: float = 30):
        if min(budget, probe_messages, holdout_messages, timeout) <= 0:
            raise ValueError("Counts and timeout must be positive")
        context = multiprocessing.get_context("spawn")
        self._connection, child = context.Pipe()
        self._process = context.Process(target=_serve, args=(
            child, profile, seed, budget, probe_messages, holdout_messages))
        self.timeout = timeout
        self._closed = False
        self._process.start()
        child.close()
        try:
            self.public = self._receive()
            p = self.public["primitives"]
            self.primitives = Primitives(tuple(p["nodes"]), tuple(p["transforms"]))
        except Exception:
            self.close()
            raise

    def _receive(self):
        if not self._connection.poll(self.timeout):
            raise TimeoutError("Evaluator response timed out")
        try:
            response = self._connection.recv()
        except EOFError as exc:
            raise RuntimeError("Evaluator stopped") from exc
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    def _call(self, request):
        if self._closed:
            raise RuntimeError("Evaluator is closed")
        self._connection.send(request)
        return self._receive()

    def probe(self, construction: Construction):
        return self._call({"op": "probe", "construction": construction_to_json(construction)})["outcome"]

    def finish(self):
        try:
            return self._call({"op": "finish"})
        finally:
            self.close()

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._connection.send({"op": "close"})
        except (BrokenPipeError, OSError):
            pass
        self._connection.close()
        self._process.join(timeout=1)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
