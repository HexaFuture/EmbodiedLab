"""Immutable public optical evidence with a separate private validity audit and private launcher state.

The public part is only what a light-gate timer itself knows: its edges, the curtain crossing, the record number and
the ids. Validity and its reasons (seating, robot contact, notch) are the evaluator's bookkeeping, kept beside it."""
from copy import deepcopy
from .protocol import payload_digest


def seal_record(eid,episode_id,measurement,start_tick,end_tick,invalid,private=None,number=None):
    public=deepcopy(measurement)|dict(evidence_id=eid,episode_id=episode_id)
    if number is not None:public['record_number']=number
    return dict(public=public,private=deepcopy(private or {}),episode_id=episode_id,start_tick=start_tick,end_tick=end_tick,
                valid=not invalid,invalid_reasons=list(invalid),sha256=payload_digest(public))
