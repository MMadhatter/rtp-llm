"""Level-3 deep-sleep collective teardown/rebuild seam -- a single interface.

Around a CUDA checkpoint (level 3 only) every cross-process collective that
holds device state incompatible with cuCheckpoint (a live NCCL/DeepEP buffer
makes Restore fail 304/801; symm-mem NVLS multicast wedges Checkpoint) must be
DESTROYED before the process is checkpointed and REBUILT on wake.

Rather than scatter that logic (teardown order here, rebuild order + gating in a
closure elsewhere), each such resource registers itself ONCE as a
:class:`CollectiveParticipant` -- a symmetric ``(teardown, rebuild)`` pair under
a name. ``BackendManager`` registers participants at startup in DEPENDENCY order
(the one another depends on first), capturing the init configs its rebuild
needs. The C++ hooks then just call :func:`run_teardown` / :func:`run_rebuild`.

Ordering is structural, not the caller's concern: teardown runs in REVERSE
registration order (a dependent is torn down before what it depends on) and
rebuild runs in registration order. So registering the process group before
DeepEP (DeepEP rendezvous runs on the PG) yields teardown ``DeepEP -> PG`` and
rebuild ``PG -> DeepEP`` automatically. Adding a new checkpoint-sensitive
collective (e.g. symm-mem) is a single register_collective() call -- no change
to the run_* drivers or the C++ boundary.
"""

import logging
import threading
from dataclasses import dataclass
from typing import Callable, List

_lock = threading.Lock()


@dataclass
class CollectiveParticipant:
    """A checkpoint-sensitive collective's destroy/rebuild pair.

    Both callables take no arguments: teardown acts on module singletons;
    rebuild closes over the init configs captured at registration time. Both
    should be safe to call when the collective was never initialized.
    """

    name: str
    teardown: Callable[[], None]
    rebuild: Callable[[], None]


_participants: List[CollectiveParticipant] = []


def register_collective(
    name: str, *, teardown: Callable[[], None], rebuild: Callable[[], None]
) -> None:
    """Register a checkpoint-sensitive collective under ``name``.

    Register in DEPENDENCY order: the collective others rely on first. Re-
    registering the same name replaces the prior entry (idempotent across
    repeated BackendManager.start), preserving position on replace.
    """
    with _lock:
        for i, p in enumerate(_participants):
            if p.name == name:
                _participants[i] = CollectiveParticipant(name, teardown, rebuild)
                return
        _participants.append(CollectiveParticipant(name, teardown, rebuild))


def clear_collectives() -> None:
    """Drop all registered participants (used by tests)."""
    with _lock:
        _participants.clear()


def registered_collectives() -> List[str]:
    with _lock:
        return [p.name for p in _participants]


def run_teardown() -> None:
    """Finalize every registered collective before checkpoint, in reverse
    registration order (dependents before their dependencies)."""
    with _lock:
        participants = list(_participants)
    for p in reversed(participants):
        logging.info("[deep-sleep] tearing down collective '%s'", p.name)
        p.teardown()


def run_rebuild() -> None:
    """Rebuild every registered collective on wake, in registration order
    (dependencies before their dependents). Raises if nothing was registered --
    a level-3 wake cannot proceed without the participants BackendManager
    registers at startup.
    """
    with _lock:
        participants = list(_participants)
    if not participants:
        raise RuntimeError(
            "no collective participants registered; cannot wake a level-3 "
            "deep-slept process (BackendManager must register them at startup)"
        )
    for p in participants:
        logging.info("[deep-sleep] rebuilding collective '%s'", p.name)
        p.rebuild()
