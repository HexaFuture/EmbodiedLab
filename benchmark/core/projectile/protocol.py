"""Pure offline projectile_landing protocol checks.

The caller must load the sealed, hash-verified private outcome. This function
never opens a service, simulates a trajectory or refits an Agent answer.
"""
import hashlib,json,math


def payload_digest(public):
    return hashlib.sha256(json.dumps(public,sort_keys=True,allow_nan=False).encode()).hexdigest()


def shot_valid(record):
    """A trial shot counts when its flight from the trigger to the curtain is clean (six-level scoring).

    The backend audits robot contact over the whole 4 s after the trigger ('robot_contact'); only contact before the
    curtain crossing matters: the robot on the ball or the plunger (or holding the ball), or the ball on the cup.
    Contact after the crossing, or a cup held during the shot, no longer voids the record. Records without the
    before-curtain flags (earlier) keep the whole-4 s rule."""
    if not isinstance(record,dict):return False
    if record.get('valid') is True:return True
    private=record.get('private') if isinstance(record.get('private'),dict) else {}
    return (record.get('invalid_reasons')==['robot_contact']
            and private.get('robot_ball_or_plunger_contact_before_curtain') is False
            and private.get('ball_cup_contact_before_curtain') is False)


def evaluate_flight(submission,records,outcome,*,episode_id,submit_tick,horizon_ticks,target_notch_m,threshold=.03,seal_valid=True):
    def finite(x): return type(x) in (int,float) and math.isfinite(x)
    gates={}
    ids=submission.get('measurement_evidence_ids') if isinstance(submission,dict) else None
    value=submission.get('predicted_distance_m') if isinstance(submission,dict) else None
    gates['valid_submission']=(finite(value) and value>0 and submission.get('unit')=='m'
        and isinstance(ids,list) and bool(ids) and all(isinstance(x,str) for x in ids)
        and len(ids)==len(set(ids)))
    gates['within_budget']=type(submit_tick) is int and 0<=submit_tick<=horizon_ticks
    gates['sealed_integrity']=bool(seal_valid)
    evidence_ok=isinstance(records,dict) and gates['valid_submission']
    target_early=False
    if isinstance(records,dict):
        for record in records.values():
            if not isinstance(record,dict) or not isinstance(record.get('public'),dict):
                evidence_ok=False
                continue
            # The latched notch is private launcher state of the record (not in the public timer memory).
            d=(record.get('private') or {}).get('notch_m')
            if finite(d) and abs(d-target_notch_m)<.0005: target_early=True
        if evidence_ok:
            for eid in ids:
                record=records.get(eid,{})
                if not isinstance(record,dict) or not isinstance(record.get('public'),dict):
                    evidence_ok=False
                    break
                public=record.get('public',{})
                events=public.get('gate_events',[])
                if not isinstance(events,list) or not all(isinstance(e,dict) for e in events) or not isinstance(public.get('landing'),dict):
                    evidence_ok=False
                    break
                try: digest_ok=record.get('sha256')==payload_digest(public)
                except (ValueError,TypeError):digest_ok=False
                evidence_ok=bool(evidence_ok and record.get('episode_id')==episode_id and digest_ok
                    and shot_valid(record) and type(record.get('end_tick')) is int
                    and record['end_tick']<submit_tick and len(events)==2
                    and [e.get('channel') for e in events]==[0,1]
                    and all(finite(e.get('t_s')) for e in events) and events[1]['t_s']>events[0]['t_s']
                    and finite((record.get('private') or {}).get('notch_m'))
                    and finite(public.get('landing',{}).get('distance_m')))
    gates['measurement_evidence_valid']=evidence_ok
    gates['prediction_before_holdout']=not target_early
    outcome=outcome if isinstance(outcome,dict) else {}
    gates['private_continuation_valid']=(outcome.get('episode_id')==episode_id
        and outcome.get('completed') is True and type(outcome.get('launch_tick')) is int
        and outcome['launch_tick']>submit_tick and outcome.get('cup_frozen') is True and outcome.get('ball_seated') is True
        and outcome.get('no_robot_contact') is True and outcome.get('range_measured_before_cup_contact') is True
        and finite(outcome.get('notch_m')) and abs(outcome['notch_m']-target_notch_m)<.0005
        and finite(outcome.get('distance_m')) and outcome['distance_m']>0)
    numerical=None
    if gates['valid_submission'] and gates['private_continuation_valid']:
        numerical=abs(value/outcome['distance_m']-1)
    gates['caught_and_retained']=(outcome.get('entered_cup') is True and outcome.get('retained_in_cup') is True)
    all_gates=all(gates.values())
    return dict(gates=gates,relative_error=numerical,task_completed=all_gates and numerical is not None and numerical<=threshold,
                reward=max(0.,1-numerical/threshold) if all_gates and numerical is not None else 0.)
