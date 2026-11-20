"""Public answer schema and trusted evidence checks. No reference-fit gate."""
import hashlib
import json
import math

MODULES = ('C100', 'C220', 'C470')
LOAD_TOLERANCE = .02


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate(answer, bench, target_load):
    if not isinstance(answer, dict) or set(answer) != {
        'experiment_id', 'load_limits', 'final_configuration', 'final_measurement', 'evidence_ids'
    } or answer['experiment_id'] != 'rectifier_supply':
        raise ValueError('Invalid answer schema')
    rows = answer['load_limits']
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError('One load limit per capacitor required')
    names = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'module_id', 'minimum_load_ohm'}:
            raise ValueError('Invalid load limit schema')
        names.append(row['module_id'])
        if not number(row['minimum_load_ohm']) or not 150 <= row['minimum_load_ohm'] <= 3300:
            raise ValueError('Invalid load limit value')
    if sorted(names) != sorted(MODULES):
        raise ValueError('Duplicate or unknown capacitor')
    config = answer['final_configuration']
    if not isinstance(config, dict) or set(config) != {'module_id', 'load_ohm'}:
        raise ValueError('Invalid final configuration')
    if config['module_id'] not in MODULES or not number(config['load_ohm']):
        raise ValueError('Invalid final configuration value')
    if abs(config['load_ohm']/target_load - 1) > LOAD_TOLERANCE:
        raise ValueError('Declared load differs from the task load by more than 2 percent')
    cap, load, mode = bench.switch
    if cap != config['module_id'] or mode != 'ON':
        raise ValueError('Actual capacitor or mode differs from final configuration')
    if max(abs(load/target_load - 1), abs(load/config['load_ohm'] - 1)) > LOAD_TOLERANCE:
        raise ValueError('Actual load differs from declared/task load by more than 2 percent')
    measurement = answer['final_measurement']
    if not isinstance(measurement, dict) or set(measurement) != {'mean_voltage_V', 'ripple_pp_V'}:
        raise ValueError('Invalid final measurement')
    if not all(number(x) and x >= 0 for x in measurement.values()):
        raise ValueError('Nonnegative finite predicted values required')
    ids = answer['evidence_ids']
    if not isinstance(ids, list) or not ids or not all(isinstance(x, str) for x in ids) or len(set(ids)) != len(ids):
        raise ValueError('Unique evidence IDs from this episode required')
    bindings = {}
    for key in ids:
        if key not in bench.records:
            raise ValueError('Evidence from this episode required')
        record = bench.records[key]
        if record['end_tick'] is None or not record.get('samples') or record['overrange']:
            raise ValueError('Complete in-range raw evidence required')
        payload = bench.record_payload(key)
        if digest(payload) != record['sha256']:
            raise ValueError('Raw evidence integrity check failed')
        if len(record['samples']) != (record['end_tick']-record['start_tick'])*50:
            raise ValueError('Evidence clock/sample count mismatch')
        # Uniform sample-clock check, not a mandated analysis window or fit.
        if any(abs(row[0]-(record['start_tick']/240+(i+1)/12000)) > 1e-8
               for i, row in enumerate(record['samples'])):
            raise ValueError('Evidence timebase mismatch')
        bindings[key] = record['sha256']
    return bindings
