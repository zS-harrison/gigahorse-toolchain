"""Calldata words of dynamic arguments printed through the argument; free-pointer
reads folded into the call whose buffer they locate.

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


def fold_pointer_reads(body) -> int:
    """`p = MEM[64]` used only as a call's buffer position (its metadata): the call
    reads MEM[64] itself, and the read's line folds into the call's.

    Only when exactly one statement uses p, after it in the same list, with
    nothing in between that could move the free pointer (only checks and local
    values reading no memory), so MEM[64] still holds p when the call runs.
    """
    from .ir import Assign, Comment, Require
    from .passes import is_free_pointer_read, map_bodies, statement_ids

    printed: dict[str, int] = {}
    for stmt in iter_statements(body):
        for expr in stmt.expressions():
            for node in _walk(expr):
                if isinstance(node, Var):
                    printed[node.name] = printed.get(node.name, 0) + 1
    folded = 0

    def quiet(stmt) -> bool:
        if isinstance(stmt, (Comment, Require)):
            return not any(_touches(e) for e in stmt.expressions())
        return (
            isinstance(stmt, Assign)
            and len(stmt.targets) == 1
            and is_local(stmt.targets[0])
            and not _touches(stmt.value)
        )

    def visit(statements):
        nonlocal folded
        index = 0
        while index < len(statements):
            stmt = statements[index]
            if not (
                isinstance(stmt, Assign)
                and len(stmt.targets) == 1
                and is_local(stmt.targets[0])
                and is_free_pointer_read(stmt.value)
                and printed.get(stmt.targets[0].name, 0) == 1  # its own definition
            ):
                index += 1
                continue
            name = stmt.targets[0].name
            user = None
            for later in statements[index + 1 :]:
                if _mentions_meta(later, name):
                    user = later
                    break
                if not quiet(later):
                    break
            if user is None or any(
                _mentions_meta(other, name)
                for other in iter_statements(statements)
                if other is not user
            ):
                index += 1
                continue
            pointer = stmt.value
            user.meta = {
                key: (
                    tuple(_replace_var(e, name, pointer) for e in value)
                    if key in {"input", "output"}
                    else value
                )
                for key, value in user.meta.items()
            }
            user.hidden = user.hidden | statement_ids(stmt)
            del statements[index]
            folded += 1
        return statements

    map_bodies(body, visit)
    return folded


def _walk(expr):
    yield expr
    for child in expr.children():
        yield from _walk(child)


def _touches(expr) -> bool:
    """Reads or writes memory, or calls anything (could move the free pointer)."""
    from .ir import Call, ExternalCallExpr, Slice

    for node in _walk(expr):
        if isinstance(node, (ExternalCallExpr, Slice)):
            return True
        if (
            isinstance(node, Index)
            and isinstance(node.base, Env)
            and node.base.name
            in {
                "MEM",
                "MEM8",
            }
        ):
            return True
        if isinstance(node, Call) and not node.name.startswith(
            ("uint", "int", "bytes", "address", "bool")
        ):
            return True
        if type(node).__name__ in {"Field", "Element", "Allocation", "AbiCodec", "MemoryBytes"}:
            return True
    return False


def _mentions_meta(stmt, name: str) -> bool:
    for key in ("input", "output"):
        for expr in stmt.meta.get(key, ()) or ():
            if any(isinstance(n, Var) and n.name == name for n in _walk(expr)):
                return True
    return False


def _replace_var(expr, name: str, replacement):
    from .types import map_expr

    return map_expr(expr, lambda n: replacement if isinstance(n, Var) and n.name == name else n)
