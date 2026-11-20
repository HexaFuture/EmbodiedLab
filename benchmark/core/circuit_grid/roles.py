"""What each instrument on the circuit board measures about the object under test (private process record only; six-level process score, record gaps "closed loop" and "across a known series resistor").

Pure graph work on one bench state-log entry (no physics, nothing the Agent sees): the netlist the bench solved
(elements between merged board nodes), the placements (to merge the board nodes again: wires, the closed switch, a
changeover's selected throw), the supply's mode. The bench writes the result into every state-log entry (`roles`),
every observation state and every logger state snapshot; the evaluation computes the same from the state log for
records made before the field existed (benchmark/evaluation/circuit_grid.py).

Objects under test (`objects` of the result, one per object):
    box      resistor_blackbox: the box resistor A-B (box_AB); rc_capacitance: the box capacitor A-B (C1);
             circuit_blackbox3: the box network on A, B, C (its fixed elements)
    cell     battery_emf: the cell on its + and - straps
    <LED>    led_planck: every LED block seated on the board (its anode + and cathode - nodes)
    <R>      resistor_blackbox only: every resistor block of printed value (the reference arm of a substitution)
Per object, over the external circuit (every element that is not the object's own, the high-impedance inputs of the
voltmeters and logger channels left out, the supply as a path while it drives):
    loops         the terminal pairs [s, t] joined by an external path: the object is in a closed circuit (charging,
                  discharging, carrying current, or settled with the circuit still closed)
    supply_loop   one of those external paths runs through the driving supply
    known_series  resistor blocks of printed value in series with the object: every external path between the two
                  terminals of a loop pair runs through the block
    instruments   per seated instrument the tags of what it measures:
        across          its pins on two terminals of the object (pair in `pairs`, e.g. "A-C")
        series          removing it breaks every path between the two terminals of a pair that it closes (any
                        instrument; an ammeter there carries the object's current)
        across_known    a voltmeter or channel with its pins on the two nodes of a known_series block (the current)
        across_ammeter  a voltmeter or channel across the object and an ammeter in series with it (the object's
                        voltage plus the ammeter's published drop)
        across_supply   a voltmeter or channel across the driving supply (with a known_series block: the object's
                        voltage as the supply voltage less the block's drop)
"""
from __future__ import annotations

KNOWN_RESISTORS = frozenset({'R1k', 'R10k', 'R100k', 'R100', 'R330', 'S01', 'S05', 'S10'})   # printed value +-0.1 %
HIGH_Z = frozenset({'V1', 'V2', 'CH1', 'CH2'})
AMMETERS = frozenset({'A'})
INSTRUMENTS = ('V1', 'V2', 'A', 'CH1', 'CH2')
SUPPLY_NODES = ('1,2', '1,3')                    # the supply's + and - straps (the cell's on battery_emf)
BOX_NODES = {'A': '5,2', 'B': '5,3', 'C': '5,4'}
SUPPLY_EDGE = '__supply'


class _Nodes:
    """Board nodes merged by wires, the closed switch and a changeover's selected throw."""

    def __init__(self, placement, switch_closed, throws):
        self.parent = {}
        for block, text in (placement or {}).items():
            nodes = _pins(text)
            if not nodes:
                continue
            kind = _kind(block)
            if kind == 'wire' or (kind == 'switch' and switch_closed):
                self.union(nodes[0], nodes[1])
            elif kind == 'changeover' and len(nodes) == 3 and (throws or {}).get(block) in ('T1', 'T2'):
                self.union(nodes[1], nodes[0] if throws[block] == 'T1' else nodes[2])

    def find(self, x):
        root = x
        while self.parent.get(root, root) != root:
            root = self.parent[root]
        while self.parent.get(x, x) != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _kind(block):
    if block == 'SW':
        return 'switch'
    if block == 'CO':
        return 'changeover'
    if block.startswith('W') and block[1:].isdigit():
        return 'wire'
    return 'block'


def _pins(text):
    """Board nodes of a placement text ('board:c,r+|c,r-' or 'board:c,rT1|c,rC|c,rT2'), else []."""
    if not isinstance(text, str) or not text.startswith('board:'):
        return []
    out = []
    for part in text[6:].split('|'):
        for suffix in ('T1', 'T2', 'C', '+', '-'):
            if part.endswith(suffix):
                out.append(part[:-len(suffix)])
                break
    return out


def _connected(edges, x, y, without=frozenset()):
    if x == y:
        return True
    seen, todo = {x}, [x]
    while todo:
        n = todo.pop()
        for name, a, b in edges:
            if name in without:
                continue
            for u, w in ((a, b), (b, a)):
                if u == n and w not in seen:
                    if w == y:
                        return True
                    seen.add(w)
                    todo.append(w)
    return False


def analyse(edges, objects, supply=None, known=KNOWN_RESISTORS, high_z=HIGH_Z, ammeters=AMMETERS):
    """Roles (module docstring). edges: [(name, node, node)] of the solved circuit (merged nodes; elements with both
    pins on one node left out); objects: {object: {'terminals': {label: node}, 'elements': [names]}}; supply: the
    supply's (+, -) nodes while it drives, else None."""
    edges = [(str(n), a, b) for n, a, b in edges if a != b]
    if supply is not None and supply[0] != supply[1]:
        edges.append((SUPPLY_EDGE, supply[0], supply[1]))
    else:
        supply = None
    pins = {n: (a, b) for n, a, b in edges if n in high_z or n in ammeters}
    out = {}
    for obj, spec in objects.items():
        own = set(spec.get('elements') or ())
        terminals = {k: v for k, v in (spec.get('terminals') or {}).items() if v is not None}
        labels = sorted(terminals)
        pairs = [(s, t) for i, s in enumerate(labels) for t in labels[i + 1:] if terminals[s] != terminals[t]]
        ext = [e for e in edges if e[0] not in own and e[0] not in high_z]
        loops = [(s, t) for s, t in pairs if _connected(ext, terminals[s], terminals[t])]
        through_supply = bool(supply) and any(_supply_on_path(ext, terminals[s], terminals[t]) for s, t in loops)
        series_known = sorted(n for n, a, b in ext if n in known and any(
            not _connected(ext, terminals[s], terminals[t], {n}) for s, t in loops))
        known_nodes = {n: {a, b} for n, a, b in ext if n in series_known}
        tags, across_pairs = {}, {}
        for inst, (p, q) in pins.items():
            mine = []
            graph = ext if inst not in high_z else ext + [(inst, p, q)]
            hit = [f'{s}-{t}' for s, t in pairs if {terminals[s], terminals[t]} == {p, q}]
            if hit:
                mine.append('across')
                across_pairs[inst] = hit[0]
            elif any(_connected(graph, terminals[s], terminals[t]) and
                     not _connected(graph, terminals[s], terminals[t], {inst}) for s, t in pairs):
                mine.append('series')
            if inst in high_z:
                if any({p, q} == nodes for nodes in known_nodes.values()):
                    mine.append('across_known')
                if _across_ammeter(p, q, terminals, pairs, pins, ext, ammeters):
                    mine.append('across_ammeter')
                if supply is not None and {p, q} == set(supply):
                    mine.append('across_supply')
            if mine:
                tags[inst] = mine
        out[obj] = {'loops': [f'{s}-{t}' for s, t in loops], 'supply_loop': bool(through_supply),
                    'known_series': series_known, 'instruments': tags, 'pairs': across_pairs}
    return {'supply': list(supply) if supply else None, 'objects': out}


def _supply_on_path(ext, x, y):
    """Some path from x to y runs through the supply: x reaches one supply node and y the other without it."""
    for name, a, b in ext:
        if name != SUPPLY_EDGE:
            continue
        others = frozenset({SUPPLY_EDGE})
        for u, w in ((a, b), (b, a)):
            if _connected(ext, x, u, others) and _connected(ext, w, y, others):
                return True
    return False


def _across_ammeter(p, q, terminals, pairs, pins, ext, ammeters):
    for s, t in pairs:
        for x, y in ((s, t), (t, s)):
            if terminals[x] not in (p, q):
                continue
            other = q if p == terminals[x] else p
            for amm in ammeters:
                if amm in pins and set(pins[amm]) == {other, terminals[y]} and \
                        not _connected(ext, terminals[x], terminals[y], {amm}):
                    return True
    return False


def objects_spec(exp_id, netlist, box_elements=None):
    """{object: {'terminals': {label: board node}, 'elements': [names]}} of an experiment (module docstring), the
    terminals as board node names (merge them with the entry's nodes)."""
    names = [str(e[0]) for e in netlist or []]
    if exp_id == 'resistor_blackbox':
        out = {'box': {'terminals': {'A': BOX_NODES['A'], 'B': BOX_NODES['B']}, 'elements': ['box_AB']}}
        for e in netlist or []:
            if str(e[0]) in KNOWN_RESISTORS:
                out[str(e[0])] = {'terminals': {'+': e[1], '-': e[2]}, 'elements': [str(e[0])]}
        return out
    if exp_id == 'rc_capacitance':
        return {'box': {'terminals': {'A': BOX_NODES['A'], 'B': BOX_NODES['B']}, 'elements': ['C1']}}
    if exp_id == 'circuit_blackbox3':
        elements = list(box_elements or [n for n in names if n in ('R1', 'R2', 'C1')])
        return {'box': {'terminals': dict(BOX_NODES), 'elements': elements}}
    if exp_id == 'battery_emf':
        return {'cell': {'terminals': {'+': SUPPLY_NODES[0], '-': SUPPLY_NODES[1]}, 'elements': ['cell']}}
    if exp_id == 'led_planck':
        return {str(e[0]): {'terminals': {'+': e[1], '-': e[2]}, 'elements': [str(e[0])]}
                for e in netlist or [] if len(e) >= 4 and e[3] == 'D'}
    return {}


def from_state(entry, exp_id, box_elements=None):
    """Roles of one state-log entry (netlist, placement, switch_closed, throws, supply mode) of an experiment."""
    nodes = _Nodes(entry.get('placement'), entry.get('switch_closed'), entry.get('throws'))
    netlist = entry.get('netlist') or []
    edges = [(str(e[0]), nodes.find(str(e[1])), nodes.find(str(e[2]))) for e in netlist]
    objects = {}
    for obj, spec in objects_spec(exp_id, netlist, box_elements).items():
        objects[obj] = {'terminals': {k: nodes.find(str(v)) for k, v in spec['terminals'].items()},
                        'elements': spec['elements']}
    mode = (entry.get('supply') or {}).get('mode')
    supply = None
    if exp_id != 'battery_emf' and mode not in (None, 'OFF', 'NONE'):
        supply = (nodes.find(SUPPLY_NODES[0]), nodes.find(SUPPLY_NODES[1]))
    return analyse(edges, objects, supply)
