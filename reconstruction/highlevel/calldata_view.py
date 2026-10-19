"""Calldata words of dynamic arguments printed through the argument.

A dynamic argument x is the calldata position of its length word: x =
msg.data[4 + 32 * i] + 4 for argument i (its definition). So

    msg.data[4 + 32 * i]                 -> x.offset   (the head word; = x - 4)
    msg.data[x.data + 32 * k + 32 * j]   -> x[k + j]   (x an array of one-word elements)

Both read exactly the word they replace (the second is `Element`, space
msg.data). Only arguments whose definition is that head word plus 4, and for
elements only arrays whose element type is one static word (not bytes[] or
tuples, whose words are offsets or members).
"""

from __future__ import annotations

from .ir import Binary, Const, Element, Env, Expr, Index, Member, Var, is_local
from .objects import _map_statement, _terms, _words
from .passes import iter_statements

ONE_WORD = ("address", "bool", "uint", "int", "bytes")


def _one_word_array(kind: str) -> bool:
    if not kind.endswith("[]"):
        return False
    element = kind[:-2]
    if any(c in element for c in "[(") or element in {"bytes", "string"}:
        return False
    if element.startswith("bytes"):
        return element[5:].isdigit()
    return element.startswith(ONE_WORD)


def _head_word(definition: Expr | None) -> int | None:
    """c for a definition msg.data[c] + 4 (either order)."""
    if not (isinstance(definition, Binary) and definition.op == "+"):
        return None
    for read, four in ((definition.left, definition.right), (definition.right, definition.left)):
        if (
            isinstance(four, Const)
            and four.value == 4
            and isinstance(read, Index)
            and isinstance(read.base, Env)
            and read.base.name == "msg.data"
            and isinstance(read.key, Const)
        ):
            return read.key.value
    return None


def _element(key: Expr, arrays: set[str], evidence) -> Element | None:
    terms = _terms(key)
    data = [
        t
        for t in terms
        if isinstance(t, Member)
        and t.name == "data"
        and isinstance(t.base, Var)
        and is_local(t.base)
        and t.base.name in arrays
    ]
    if len(data) != 1:
        return None
    rest = list(terms)
    rest.remove(data[0])
    constant = sum(t.value for t in rest if isinstance(t, Const))
    scaled = [t for t in rest if not isinstance(t, Const)]
    if constant % 32 or not 0 <= constant < 1 << 64 or len(scaled) > 1:
        return None
    if scaled:
        index = _words(scaled[0])
        if index is None:
            return None
        if constant:
            index = Binary("+", index, Const(constant // 32))
    else:
        index = Const(constant // 32)
    base = data[0].base
    assert isinstance(base, Var)
    return Element(base, index, "msg.data", evidence=evidence)


def calldata_accessors(arguments, body) -> int:
    """Rewrite a public function's calldata words; returns how many."""
    offsets: dict[int, Var] = {}
    arrays: set[str] = set()
    for argument in arguments:
        position = _head_word(argument.definition)
        if position is None:
            continue
        offsets[position] = Var(argument.name, f"arg:{argument.name}")
        if _one_word_array(argument.type):
            arrays.add(argument.name)
    if not offsets:
        return 0
    count = 0

    def fn(expr: Expr) -> Expr:
        nonlocal count
        if not (isinstance(expr, Index) and isinstance(expr.base, Env)):
            return expr
        if expr.base.name != "msg.data":
            return expr
        if isinstance(expr.key, Const) and expr.key.value in offsets:
            count += 1
            return Member(offsets[expr.key.value], "offset", evidence=expr.all_evidence())
        element = _element(expr.key, arrays, expr.all_evidence()) if arrays else None
        if element is not None:
            count += 1
            return element
        return expr

    for stmt in iter_statements(body):
        _map_statement(stmt, fn)
    return count
