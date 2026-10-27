"""Process-score rules, one module per kit.

LADDERS: exp_id -> benchmark.evaluation.process.Ladder, the six-level detectors (L1-L4) of each experiment and its
extra tidy-up gates; L5 and L6 are computed in benchmark/evaluation/process.py for every experiment. A family owns
its module and the LADDERS entries in it.
RULES: exp_id -> legacy rule(run) -> {"E1", "E2", "M1", "M2": Hit} (scorer embodiedlab-process/1), kept for the
provisional ladders (process.from_legacy) and their unit tests until a family replaces them.

A rule reads only the run's process record (benchmark.evaluation.process.ProcessRun): the Bench's frame logs,
instrument records, state and action logs, evidence and episode.json (calls, exports). It never reads the instance
truth or the submission.
"""
from benchmark.evaluation.process_rules import chemistry, circuits, mechanics, optics, thermal

RULES = {**mechanics.RULES, **circuits.RULES, **optics.RULES, **thermal.RULES, **chemistry.RULES}
LADDERS = {**mechanics.LADDERS, **circuits.LADDERS, **optics.LADDERS, **thermal.LADDERS, **chemistry.LADDERS}
