"""Strict data-only protocol shared by external proposal and evaluator adapters."""
from .ideas import Construction, Primitives


def construction_to_json(value: Construction) -> dict:
    return {"path": list(value.path), "transforms": list(value.transforms),
            "repetitions": value.repetitions, "flip": value.flip}


def construction_from_json(value: object, primitives: Primitives) -> Construction:
    if not isinstance(value, dict) or set(value) != {"path", "transforms", "repetitions", "flip"}:
        raise ValueError("Expected exactly path, transforms, repetitions and flip")
    path, transforms = value["path"], value["transforms"]
    repetitions, flip = value["repetitions"], value["flip"]
    if (type(path) is not list or type(transforms) is not list
            or any(type(x) is not str or x not in primitives.nodes for x in path)
            or any(type(x) is not str or x not in primitives.transforms for x in transforms)
            or type(repetitions) is not int or repetitions < 1 or repetitions % 2 != 1
            or type(flip) is not bool):
        raise ValueError("Construction contains invalid primitives or values")
    return Construction(tuple(path), tuple(transforms), repetitions, flip)
