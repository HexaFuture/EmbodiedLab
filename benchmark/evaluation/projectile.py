"""Offline adapter: score against the sealed physical flight, never a formula."""
import json
from benchmark.core.projectile.protocol import evaluate_flight


def evaluate_projectile(spec,sealed,instance):
    ep=sealed.episode
    binding=(not sealed.manifest_problems and 'outcome.json' in sealed.manifest
        and ep.get('instance_id')==instance['instance_id'] and ep.get('exp_id')==spec.id
        and ep.get('spec_digest')==instance['spec_digest']==spec.digest())
    try:outcome=json.loads((sealed.root/'outcome.json').read_text())
    except (OSError,ValueError):outcome={};binding=False
    report=evaluate_flight(sealed.submission,sealed.evidence,outcome,episode_id=ep.get('episode_id'),
        submit_tick=ep.get('submit_tick'),horizon_ticks=ep.get('horizon_ticks',0),
        target_notch_m=instance['hidden']['target_notch_m'],threshold=spec.scoring['threshold'],seal_valid=binding)
    from benchmark.evaluation.gates import Context, valid_submission
    valid,_=valid_submission(Context(spec=spec,hidden=instance['hidden'],sealed=sealed))
    report['gates']['valid_submission'] &= bool(valid)
    report['gates']['single_submission']=ep.get('submission_attempts')==1
    report['gates']['budget_bound']=ep.get('horizon_ticks')==int(spec.budget['sim_s']*ep.get('physics_hz',0))
    order=spec.gate_params('bench_in_order')
    if order is not None:
        from benchmark.evaluation.gates import bench_in_order
        report['gates']['bench_in_order'],order_detail=bench_in_order(Context(spec=spec,hidden=instance['hidden'],sealed=sealed),**order)
    G=all(report['gates'].values());relative=report['relative_error'];e=None if relative is None else relative/spec.scoring['threshold']
    from benchmark.evaluation.evaluate import quantity
    # the truth is the landing distance of the sealed flight
    quantities=None if relative is None else [quantity('distance_m',sealed.submission['predicted_distance_m'],
        outcome['distance_m'],'rel_error',spec.scoring['threshold'],relative)]
    return dict(schema_version='embodiedlab.evaluation/v2',evaluator='embodiedlab-projectile-eval/1',
        exp_id=spec.id,spec_digest=spec.digest(),instance_id=instance['instance_id'],split=instance.get('split'),
        episode_id=ep.get('episode_id'),gates=report['gates'],
        gate_details={'bench_in_order':order_detail} if order is not None else {},G=G,
        threshold=spec.scoring['threshold'],metric=spec.scoring['metric'],relative_errors=[relative] if relative is not None else None,
        e_rel=relative,e=e,quantities=quantities,reward=max(0.,1-e) if G and e is not None else 0.,task_completed=bool(G and e is not None and e<=1),
        sim_s_used=ep['submit_tick']/ep['physics_hz'],diagnostics={'private_outcome':outcome})
