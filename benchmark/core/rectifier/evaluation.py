"""Answer-to-sealed-truth scoring; supplied by the trusted evaluator only."""
import math


def evaluate(answer, limits_truth, waveform, *, evidence_valid, budget_valid):
    errors={}
    for row in answer['load_limits']:
        true=limits_truth[row['module_id']]
        errors[row['module_id']+'_load']=abs(row['minimum_load_ohm']/true-1)/.05
    mean=waveform['mean_voltage_V'];ripple=waveform['ripple_pp_V']
    predicted=answer['final_measurement']
    errors['mean']=abs(predicted['mean_voltage_V']/mean-1)/.03 if mean>0 else math.inf
    errors['ripple']=abs(predicted['ripple_pp_V']-ripple)/max(.03,.05*ripple)
    g=bool(evidence_valid and budget_valid and waveform['configuration_valid'] and mean>=8.5 and ripple<=.8)
    e=max(errors.values())
    return dict(G=g,normalized_errors=errors,e=e,task_completed=g and e<=1)
