"""Restricted arithmetic for experiment specifications.

Specification expressions (distribution bounds, derived hidden values,
constraints, gate parameters) are evaluated by the trusted benchmark host,
never inside an Agent sandbox. Only literals, names bound by the caller,
arithmetic, comparisons, boolean logic, conditional expressions, subscripts
and the whitelisted functions below are accepted.
"""
from __future__ import annotations

import ast
import math
import operator


def mode_freqs(omega_a: float, omega_b: float, coupling: float) -> tuple[float, float]:
    """Undamped normal-mode angular frequencies of the equal-mass coupled pair.

    Eigenvalues of [[omega_a^2 + c, -c], [-c, omega_b^2 + c]], ascending.
    Same closed form as benchmark/core/coupled_oscillator/dynamics.py.
    """
    a = omega_a * omega_a + coupling
    d = omega_b * omega_b + coupling
    delta = math.hypot(a - d, 2 * coupling)
    return (math.sqrt((a + d - delta) / 2), math.sqrt((a + d + delta) / 2))


def _dye(name: str):
    """Effective (slit-convolved) spectrum of a chemistry-kit dye at its literature scale (benchmark/core/chem_solver)."""
    if name not in _DYES:
        import importlib.util
        from pathlib import Path
        path = Path(__file__).resolve().parents[1] / "core" / "chem_solver.py"
        spec = importlib.util.spec_from_file_location("embodiedlab_chem_solver", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _DYES[name] = module.Dye(name)
    return _DYES[name]


_DYES: dict = {}


def dye_lambda_max(dye: str) -> float:
    """Wavelength of the effective absorption maximum, to 1 nm."""
    return int(round(_dye(dye).lambda_max_nm))


def dye_eps(dye: str, wavelength_nm: float) -> float:
    """Effective molar absorptivity (L mol^-1 cm^-1) at a wavelength, before the batch factor."""
    return _dye(dye).eps_at(wavelength_nm)


FUNCTIONS = {
    "sqrt": math.sqrt, "log": math.log, "exp": math.exp, "abs": abs,
    "min": min, "max": max, "pi": math.pi, "mode_freqs": mode_freqs, "round": round,
    "dye_lambda_max": dye_lambda_max, "dye_eps": dye_eps,
}

_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
           ast.FloorDiv: operator.floordiv}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Not: operator.not_}
_COMPARE = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
            ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne}


class ExpressionError(ValueError):
    pass


def names(source: str) -> set[str]:
    """Free names referenced by an expression (excluding whitelisted functions)."""
    tree = _parse(source)
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} - set(FUNCTIONS)


def evaluate(source, scope: dict):
    """Evaluate a spec expression; non-strings are returned unchanged."""
    if not isinstance(source, str):
        return source
    return _eval(_parse(source).body, scope, source)


def _parse(source: str) -> ast.Expression:
    if not isinstance(source, str) or len(source) > 500:
        raise ExpressionError("expression must be a string of at most 500 characters")
    try:
        return ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"invalid expression {source!r}: {exc.msg}") from None


def _eval(node, scope, source):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool, str)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in scope:
            return scope[node.id]
        if node.id in FUNCTIONS:
            return FUNCTIONS[node.id]
        raise ExpressionError(f"unknown name {node.id!r} in {source!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _BINARY[type(node.op)](_eval(node.left, scope, source), _eval(node.right, scope, source))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand, scope, source))
    if isinstance(node, ast.BoolOp):
        values = (_eval(v, scope, source) for v in node.values)
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.Compare):
        left = _eval(node.left, scope, source)
        for op, comparator in zip(node.ops, node.comparators):
            if type(op) not in _COMPARE:
                raise ExpressionError(f"unsupported comparison in {source!r}")
            right = _eval(comparator, scope, source)
            if not _COMPARE[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        branch = node.body if _eval(node.test, scope, source) else node.orelse
        return _eval(branch, scope, source)
    if isinstance(node, ast.Subscript):
        return _eval(node.value, scope, source)[_eval(node.slice, scope, source)]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_eval(v, scope, source) for v in node.elts]
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in FUNCTIONS and not node.keywords):
        function = FUNCTIONS[node.func.id]
        if not callable(function):
            raise ExpressionError(f"{node.func.id!r} is not callable")
        return function(*(_eval(a, scope, source) for a in node.args))
    raise ExpressionError(f"unsupported syntax in {source!r}")
