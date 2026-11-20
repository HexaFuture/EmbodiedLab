"""circuit_blackbox3 protocol checks. Numerical accuracy is scored against truth, not a refit."""
import csv
import io
import math


def parameter_errors(hidden, submission):
    if submission['topology'] != hidden['topology']:
        return [1.0]  # categorical failure; unused sampled values are never scored
    return [abs(value / hidden[key] - 1) for key, value in submission['parameters'].items()]


def topology_correct(ctx):
    ok = ctx.submission_valid and ctx.submission['topology'] == ctx.hidden['topology']
    return bool(ok), {'matches': bool(ok)}


def network_evidence(ctx, min_window_s=5.0, min_samples=30):
    if not ctx.submission_valid:
        return False, {'problems': ['invalid submission']}
    problems, summaries = [], []
    for eid in ctx.submission['measurement_evidence_ids']:
        record, problem = ctx.evidence(eid)
        if problem:
            problems.append(problem)
            continue
        if record.get('kind') != 'circuit_logger' or record.get('status') != 'complete':
            problems.append('expected complete circuit logger record')
            continue
        try:
            rows = list(csv.DictReader(io.StringIO(record['csv'])))
            segments, previous, last_time = [], None, -math.inf
            for row in rows:
                t = float(row['t_s'])
                if not math.isfinite(t) or t <= last_time:
                    raise ValueError('non-monotone timestamps')
                last_time = t
                if not all(math.isfinite(float(row[c])) for c in ['ch1_V', 'ch2_V']):
                    raise ValueError('non-finite voltage')
                valid = (row['status'] == 'ok' and row['mode'] in ('CHARGE', 'DISCHARGE')
                         and row['pair'] in ('A-B', 'A-C', 'B-C')
                         and str(row['rs']) in ('1000', '10000', '100000'))
                key = (row['pair'], row['rs'], row['mode']) if valid else None
                if key != previous or not segments:
                    segments.append((key, []))
                segments[-1][1].append(t)
                previous = key
            usable = [(k, ts) for k, ts in segments if k and len(ts) >= min_samples
                      and ts[-1] - ts[0] >= min_window_s - 1e-6]
            if not usable:
                problems.append(f'{eid}: no sufficiently long valid driven segment')
            summaries.append({'evidence_id': eid, 'valid_segments': len(usable)})
        except (ValueError, KeyError, TypeError) as exc:
            problems.append(f'{eid}: invalid logger payload: {exc}')
    return not problems, {'problems': problems, 'records': summaries}
