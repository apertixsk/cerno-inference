"""The two example dataclasses of the decider recipe, as used by the inference path: one context plus one or more typed
questions, each with a fixed option list (large label sets are rendered in chunks of MAX_OPTIONS at formatting time)."""
from dataclasses import dataclass

MAX_OPTIONS = 10


@dataclass
class Q:
    text: str
    options: list          # canonical option strings
    gold: int              # index into options (or -1 for unknown)


@dataclass
class Example:
    context: str
    qs: list               # list[Q]
    task: str
    image: bytes = None    # optional PNG bytes (vision variants); use getattr(e, "image", None)
