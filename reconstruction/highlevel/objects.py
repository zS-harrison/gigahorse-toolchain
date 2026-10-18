"""Memory objects: allocations at the free pointer printed as values.

solc allocates memory by reading the free pointer and moving it past the new
object (`allocate_memory`):

    v0 = MEM[64]; MEM[64] = v0 + 224;                    -> v0 = new struct(7)
    v2 = MEM[64]; MEM[64] = v2 + round(n) + 32;
        MEM[v2] = n;                                     -> v2 = new bytes(n)
        [CALLDATACOPY(v2 + 32, x.data, x.length);
         MEM[v2 + 32 + x.length] = 0;]                   -> v2 = x  (copied to memory)

Words at constant offsets from such a pointer print as fields (`v0.word3`,
`v2.length`); a struct whose words are all written right after the
allocation, with values that read no memory, prints as `struct(a, b, ...)`.
Where the analysis folded the pointer to a constant (`MEM[64] = 352`,
`MEM[160] = x`), the object's base is assumed to be solc's initial free
pointer 0x80; the interpreter checks that when it executes the allocation
(rule assumes-folded-free-pointer).

This runs last, after every memory analysis, and only renames: a Field reads
and writes exactly MEM[base + offset], and an Allocation performs exactly the
statements it replaced, in their order.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .abi_codec import (
    NOT31,
    Form,
    Tracker,
    _pointer_read,
    _pointer_update,
    _store,
    _walk,
    add,
    constant_of,
)
from .ir import (
    AbiCodec,
    Allocation,
    Assign,
    Binary,
    Call,
    Comment,
    Const,
    Element,
    Emit,
    Env,
    Expr,
    ExprStmt,
    Field,
    Halt,
    If,
    Index,
    Member,
    MemoryBytes,
    ParallelAssign,
    Require,
    Return,
    Revert,
    Slice,
    Stmt,
    Var,
    While,
    annotate,
    is_local,
)
from .passes import (
    FunctionState,
    has_effect,
    iter_expr,
    iter_statements,
    statement_ids,
    substitute,
)
from .types import map_expr

INITIAL_POINTER = 0x80  # solc's free pointer before any allocation
MAX_WORDS = 64
WINDOW = 6


@dataclass
class _Object:
    name: str
    kind: str
    size: int | None  # struct: bytes
    assumed_base: int | None
    statement: Assign
    # The base address when it is a constant (MEM[64] was set to one, or the
    # folded initial pointer is assumed): constant addresses can name fields.
    constant_base: int | None = None


def _reads_memory(expr: Expr) -> bool:
    for node in _walk(expr):
        if isinstance(node, (Field, Element, Slice, MemoryBytes)):
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
        if isinstance(node, Env) and node.name in {"MEM", "MEM8"}:
            return True
    return False


def _quiet(stmt: Stmt) -> bool:
    """No effect, no memory access: may move across an allocation."""
    if isinstance(stmt, Assign):
        return (
            len(stmt.targets) == 1
            and is_local(stmt.targets[0])
            and not has_effect(stmt.value)
            and not _reads_memory(stmt.value)
        )
    if isinstance(stmt, Require):
        exprs = [stmt.condition] + ([stmt.payload] if stmt.payload is not None else [])
        return not any(has_effect(e) or _reads_memory(e) for e in exprs)
    return False


def _mentions(expr: Expr, name: str) -> bool:
    return any(isinstance(node, Var) and node.name == name for node in _walk(expr))


def _rounded_forms(length: Expr) -> list[Expr]:
    """solc's byte-array allocation sizes: round_up(length) + 32 (three spellings)."""
    plus31 = Binary("+", length, Const(31))
    masked = Binary("&", plus31, Const(NOT31))
    return [
        Binary("+", Binary("*", Binary("/", plus31, Const(32)), Const(32)), Const(32)),
        Binary("+", masked, Const(32)),
        Binary("&", Binary("+", masked, Const(63)), Const(NOT31)),
        Binary("&", Binary("+", length, Const(63)), Const(NOT31)),
    ]


def _keeps_pointer(stmt: Stmt) -> bool:
    """Cannot move the free pointer (memory reads and word stores are fine)."""
    if isinstance(stmt, Comment) or _store(stmt) is not None:
        return True
    if isinstance(stmt, Assign):
        return len(stmt.targets) == 1 and is_local(stmt.targets[0]) and not has_effect(stmt.value)
    if isinstance(stmt, ParallelAssign):
        return all(is_local(x) for x in stmt.targets) and not any(
            has_effect(v) for v in stmt.values
        )
    if isinstance(stmt, Require):
        return not has_effect(stmt.condition)
    if isinstance(stmt, ExprStmt) and isinstance(stmt.expr, Call):
        return stmt.expr.name in {"CALLDATACOPY", "CODECOPY", "RETURNDATACOPY", "MCOPY"}
    if isinstance(stmt, If):
        return not has_effect(stmt.condition) and all(
            _keeps_pointer(s) for body in stmt.blocks() for s in body
        )
    return False


def _array_forms(length: Expr) -> list[Expr]:
    """solc's array allocation sizes: length * 32 + 32 (via-IR: rounded up, the same)."""
    words = [Binary("*", length, Const(32)), Binary("<<", length, Const(5))]
    return [Binary("+", w, Const(32)) for w in words] + [
        Binary("&", Binary("+", w, Const(63)), Const(NOT31)) for w in words
    ]


def _match(
    statements: list[Stmt],
    index: int,
    tracker: Tracker,
    assignments: Counter[str],
    entry: bool,
) -> tuple[list[Stmt], int, _Object, Tracker] | None:
    first = statements[index]
    pointer = _pointer_read(first)
    if pointer is None or assignments[pointer.name] != 1:
        return None
    t = tracker.copy()
    t.observe(first)
    base = t.canon(pointer)
    kept: list[Stmt] = []
    early: Stmt | None = None  # a word stored at the base before the pointer moves
    position = index + 1
    while position < len(statements) and position <= index + WINDOW:
        stmt = statements[position]
        if _pointer_update(stmt) is not None:
            break
        stored = _store(stmt)
        if early is None and stored is not None and t.canon(stored[0]) == base:
            early = stmt
        elif not _quiet(stmt):
            break
        else:
            kept.append(stmt)
        t.observe(stmt)
        position += 1
    if position >= len(statements):
        return None
    update_stmt = statements[position]
    update = _pointer_update(update_stmt)
    if update is None:
        return None
    # The quiet statements move after the allocation: its end cannot read the
    # locals they assign (e.g. a common subexpression shared with the size check).
    definitions = {
        stmt.targets[0].name: stmt.value
        for stmt in kept
        if isinstance(stmt, Assign) and len(stmt.targets) == 1 and is_local(stmt.targets[0])
    }
    if any(_mentions(update, name) for name in definitions):
        update = substitute(update, definitions)
    end = t.canon(update)
    t.observe(update_stmt)
    consumed = [update_stmt]
    after = position + 1
    size = constant_of(add(end, base, -1))
    assumed = None
    # The pointer is still the one the dispatcher set (0x80) only at a public
    # entry point before anything could move it; checked when executed.
    folded = entry and base[0] == "fp" and base in t.bases
    if size is None and folded and constant_of(end) is not None:
        assumed = INITIAL_POINTER
        size = (constant_of(end) or 0) - INITIAL_POINTER
    if size is not None:
        if not (0 < size <= 32 * MAX_WORDS and size % 32 == 0):
            return None
        node = Allocation("struct", update, words=size // 32, assumed_base=assumed)
        obj = _Object(pointer.name, "struct", size, assumed, first)
        if early is not None:
            kept.insert(0, early)  # a word-0 write: it stays, after the allocation
    else:
        # new bytes(n) / new T[](n): a length word at the base, and a size from it.
        length_stmt = early
        if length_stmt is None and after < len(statements):
            stored = _store(statements[after])
            if stored is not None and t.canon(stored[0]) == base:
                length_stmt = statements[after]
                t.observe(length_stmt)
                after += 1
        if length_stmt is None:
            return None
        stored = _store(length_stmt)
        assert stored is not None
        length = stored[1]
        if has_effect(length) or _reads_memory(length) or _mentions(length, pointer.name):
            return None
        consumed.append(length_stmt)
        added = add(end, base, -1)
        if any(t.canon(form) == added for form in _rounded_forms(length)):
            kind = "bytes"
        elif any(t.canon(form) == added for form in _array_forms(length)):
            kind = "array"
        else:
            return None
        node = Allocation(kind, update, length=length)
        obj = _Object(pointer.name, kind, None, None, first)
        fill = _fill(statements, after, t, base, kind, length, folded)
        if fill is not None:
            moved, (space, source, copied, pad), consumed_fill, after, assumed = fill
            kept.extend(moved)
            consumed.extend(consumed_fill)
            for stmt in [*moved, *consumed_fill]:
                t.observe(stmt)
            node = Allocation(
                kind,
                update,
                length=length,
                copy=(source, copied),
                pad=pad,
                assumed_base=assumed,
                copy_space=space,
            )
            obj.assumed_base = assumed
    rules = {"memory-object"} | ({"assumes-folded-free-pointer"} if node.assumed_base else set())
    allocation = Assign(
        (pointer,),
        annotate(node, rules=frozenset(rules)),
        evidence=first.evidence,
        hidden=first.hidden.union(*(statement_ids(s) for s in consumed)),
        rules=first.rules | frozenset(rules),
        origin=first.origin,
    )
    if node.copy is not None:
        allocation.note = {
            "calldata": f"calldata {'array' if node.kind == 'array' else 'bytes'} copied to memory",
            "returndata": "returned data copied to memory",
        }[node.copy_space]
    obj.statement = allocation
    if obj.assumed_base is not None:
        obj.constant_base = obj.assumed_base
    elif base[0] == "const":
        obj.constant_base = base[1]
    # t: the facts after the replaced statements (the pointer is now `end`).
    return [allocation, *kept], after, obj, t


def _fill(
    statements: list[Stmt],
    index: int,
    t: Tracker,
    base: Form,
    kind: str,
    length: Expr,
    folded: bool,
):
    """The data copied in right after the length word (after quiet code):
    CALLDATACOPY(base + 32, x.data, size) for a calldata argument, or
    RETURNDATACOPY(base + 32, 0, returndata.length); then maybe a zero word.
    (Zeroing `new T[](n)` from past the end of calldata is already removed as
    a no-op on fresh memory by the memory passes.)"""
    moved: list[Stmt] = []
    position = index
    while position < len(statements) and _quiet(statements[position]) and len(moved) < WINDOW:
        moved.append(statements[position])
        position += 1
    if position >= len(statements):
        return None
    stmt = statements[position]
    if not (
        isinstance(stmt, ExprStmt)
        and isinstance(stmt.expr, Call)
        and stmt.expr.name in {"CALLDATACOPY", "RETURNDATACOPY"}
        and len(stmt.expr.args) == 3
    ):
        return None
    destination, source, size = stmt.expr.args
    sizes = (
        [length]
        if kind == "bytes"
        else [Binary("*", length, Const(32)), Binary("<<", length, Const(5))]
    )
    if not any(t.canon(size) == t.canon(s) for s in sizes):
        return None
    if stmt.expr.name == "RETURNDATACOPY":
        if not (
            kind == "bytes"
            and isinstance(source, Const)
            and source.value == 0
            and isinstance(length, Member)
            and isinstance(length.base, Env)
            and length.base.name == "returndata"
        ):
            return None
        space = "returndata"
    elif isinstance(source, Member) and source.name == "data" and is_local(source.base):
        space = "calldata"
    else:
        return None
    assumed = None
    data = add(base, ("const", 32))
    if t.canon(destination) != data:
        constant = constant_of(t.canon(destination))
        if not (folded and constant == INITIAL_POINTER + 32):
            return None
        assumed = INITIAL_POINTER
        data = ("const", constant)
    consumed = [stmt]
    pad = False
    following = position + 1
    if following < len(statements):
        stored = _store(statements[following])
        if (
            stored is not None
            and isinstance(stored[1], Const)
            and stored[1].value == 0
            and t.canon(stored[0]) == add(data, t.canon(size))
        ):
            consumed.append(statements[following])
            pad = True
            following += 1
    return moved, (space, source, size, pad), consumed, following, assumed


def _find(
    statements: list[Stmt],
    tracker: Tracker,
    assignments: Counter[str],
    objects: dict[str, _Object],
    scopes: dict[str, list[Stmt]],
    entry: bool,
) -> list[Stmt]:
    result: list[Stmt] = []
    index = 0
    while index < len(statements):
        match = _match(statements, index, tracker, assignments, entry)
        if match is not None:
            replacement, index, obj, after = match
            objects[obj.name] = obj
            tracker.adopt(after)
            entry = False
            # Constant addresses are renamed only where the allocation dominates.
            scopes[obj.name] = statements[index:]
            result.extend(replacement)
            continue
        stmt = statements[index]
        for body in stmt.blocks():
            body[:] = _find(body, tracker.copy(), assignments, objects, scopes, entry)
        if isinstance(stmt, If) and _keeps_pointer(stmt):
            # Neither branch moves the pointer: only the locals they assign change.
            for nested in iter_statements([stmt]):
                for target in getattr(nested, "targets", ()):
                    if is_local(target):
                        tracker.env.pop(target.name, None)
        else:
            tracker.observe(stmt)
        entry = entry and _keeps_pointer(stmt)
        result.append(stmt)
        index += 1
    return result


def _field(obj: _Object, offset: int, evidence) -> Field | None:
    if obj.kind in {"bytes", "array"} and offset == 0:
        return Field(Var(obj.name, None), 0, "length", evidence=evidence)
    if obj.kind == "array" and offset >= 32 and offset % 32 == 0:
        return Field(Var(obj.name, None), offset, f"[{offset // 32 - 1}]", evidence=evidence)
    if obj.kind in {"bytes", "array"}:
        return None
    assert obj.size is not None
    if 0 <= offset < obj.size and offset % 32 == 0:
        return Field(Var(obj.name, None), offset, f"word{offset // 32}", evidence=evidence)
    return None


def _relative(key: Expr) -> tuple[str, int] | None:
    if isinstance(key, Var) and is_local(key):
        return key.name, 0
    if isinstance(key, Binary) and key.op == "+":
        for a, b in ((key.left, key.right), (key.right, key.left)):
            if isinstance(a, Var) and is_local(a) and isinstance(b, Const):
                return a.name, b.value
    return None


def _terms(expr: Expr) -> list[Expr]:
    if isinstance(expr, Binary) and expr.op == "+":
        return _terms(expr.left) + _terms(expr.right)
    return [expr]


def _words(expr: Expr) -> Expr | None:
    """X for X * 32, 32 * X or X << 5."""
    if isinstance(expr, Binary) and isinstance(expr.right, Const):
        if (expr.op == "*" and expr.right.value == 32) or (
            expr.op == "<<" and expr.right.value == 5
        ):
            return expr.left
    if isinstance(expr, Binary) and expr.op == "*" and isinstance(expr.left, Const):
        return expr.right if expr.left.value == 32 else None
    return None


def _element(key: Expr, objects: dict[str, _Object], evidence) -> Element | None:
    """MEM[p + 32 * X + 32 * (k + 1)] -> p[X + k] for a memory array p (k >= 0)."""
    terms = _terms(key)
    arrays = [
        t
        for t in terms
        if isinstance(t, Var) and is_local(t) and t.name in objects
        if objects[t.name].kind == "array"
    ]
    if len(arrays) != 1:
        return None
    rest = list(terms)
    rest.remove(arrays[0])
    constant = sum(t.value for t in rest if isinstance(t, Const))
    scaled = [t for t in rest if not isinstance(t, Const)]
    index = _words(scaled[0]) if len(scaled) == 1 else None
    if index is None or constant < 32 or constant % 32 or constant >= 1 << 64:
        return None
    if constant > 32:
        index = Binary("+", index, Const(constant // 32 - 1))
    return Element(Var(arrays[0].name, None), index, evidence=evidence)


def _map_statement(stmt: Stmt, fn) -> None:
    """Rewrite a statement's own expressions (not nested bodies); locals stay bound."""
    if isinstance(stmt, Assign):
        stmt.targets = tuple(t if is_local(t) else map_expr(t, fn) for t in stmt.targets)
        stmt.value = map_expr(stmt.value, fn)
    elif isinstance(stmt, ParallelAssign):
        stmt.values = tuple(map_expr(v, fn) for v in stmt.values)
    elif isinstance(stmt, ExprStmt):
        stmt.expr = map_expr(stmt.expr, fn)
    elif isinstance(stmt, Require):
        stmt.condition = map_expr(stmt.condition, fn)
        if stmt.payload is not None:
            stmt.payload = map_expr(stmt.payload, fn)
    elif isinstance(stmt, (If, While)) and stmt.condition is not None:
        stmt.condition = map_expr(stmt.condition, fn)
    elif isinstance(stmt, Return):
        stmt.values = tuple(map_expr(v, fn) for v in stmt.values)
        if stmt.raw is not None:
            stmt.raw = map_expr(stmt.raw, fn)
    elif isinstance(stmt, Revert) and stmt.payload is not None:
        stmt.payload = map_expr(stmt.payload, fn)
    elif isinstance(stmt, Emit):
        stmt.args = tuple(map_expr(a, fn) for a in stmt.args)
    elif isinstance(stmt, Halt) and stmt.operand is not None:
        stmt.operand = map_expr(stmt.operand, fn)


def _rename(statements: list[Stmt], objects: dict[str, _Object], constants: dict[int, tuple]):
    """MEM[p + c] -> p.wordN everywhere; MEM[A] -> p.wordN for constant-base objects in scope."""

    def fn(expr: Expr) -> Expr:
        if not (isinstance(expr, Index) and isinstance(expr.base, Env) and expr.base.name == "MEM"):
            return expr
        relative = _relative(expr.key)
        if relative is not None and relative[0] in objects:
            obj = objects[relative[0]]
            field = _field(obj, relative[1], expr.all_evidence())
            return field if field is not None else expr
        element = _element(expr.key, objects, expr.all_evidence())
        if element is not None:
            return element
        if isinstance(expr.key, Const) and expr.key.value in constants:
            obj, offset = constants[expr.key.value]
            field = _field(obj, offset, expr.all_evidence())
            return field if field is not None else expr
        return expr

    for stmt in iter_statements(statements):
        _map_statement(stmt, fn)


def _literals(statements: list[Stmt], objects: dict[str, _Object]) -> list[Stmt]:
    """`p = new struct(k)` followed by writes of all k words: `p = struct(values)`."""
    result: list[Stmt] = []
    index = 0
    while index < len(statements):
        stmt = statements[index]
        for body in stmt.blocks():
            body[:] = _literals(body, objects)
        obj = None
        if isinstance(stmt, Assign) and len(stmt.targets) == 1 and is_local(stmt.targets[0]):
            obj = objects.get(stmt.targets[0].name)
        node = stmt.value if isinstance(stmt, Assign) else None
        if obj is None or not isinstance(node, Allocation) or node.kind != "struct":
            result.append(stmt)
            index += 1
            continue
        assert node.words is not None
        values: dict[int, Expr] = {}
        moved: list[Stmt] = []
        writes: list[Stmt] = []
        position = index + 1
        while position < len(statements) and len(values) < node.words:
            other = statements[position]
            target = other.targets[0] if isinstance(other, Assign) and other.targets else None
            if (
                isinstance(target, Field)
                and isinstance(target.base, Var)
                and target.base.name == obj.name
                and len(other.targets) == 1
                and target.offset // 32 not in values
                and not has_effect(other.value)
                and not _reads_memory(other.value)
                and not _mentions(other.value, obj.name)
            ):
                values[target.offset // 32] = other.value
                writes.append(other)
            elif _quiet(other) and not any(_mentions(e, obj.name) for e in other.expressions()):
                moved.append(other)
            else:
                break
            position += 1
        if len(values) != node.words or node.words > 16:
            result.append(stmt)
            index += 1
            continue
        ordered = tuple(values[i] for i in range(node.words))
        text = _constant_bytes(ordered)
        literal = Allocation(
            "struct",
            node.end,
            words=node.words,
            values=ordered,
            assumed_base=node.assumed_base,
            text=text,
            evidence=node.evidence,
            rules=node.rules | {"string-literal" if text is not None else "struct-literal"},
        )
        if text is not None:
            obj.kind = "bytes-literal"
        stmt.value = literal
        stmt.hidden = stmt.hidden.union(*(statement_ids(w) for w in writes))
        result.extend(moved)
        result.append(stmt)
        index = position
    return result


def _constant_bytes(values: tuple[Expr, ...]) -> bytes | None:
    """The bytes of `length, data...` words laid out exactly as a bytes/string constant."""
    if len(values) < 2 or not all(isinstance(v, Const) for v in values):
        return None
    length = values[0].value  # type: ignore[attr-defined]
    if not (0 < length <= 32 * (len(values) - 1)) or (length + 31) // 32 != len(values) - 1:
        return None
    data = b"".join(v.value.to_bytes(32, "big") for v in values[1:])  # type: ignore[attr-defined]
    if any(data[length:]):
        return None
    return data[:length]


def _relabel_lengths(statements: list[Stmt], names: set[str]) -> None:
    """word0 of a bytes constant is its length."""

    def fn(expr: Expr) -> Expr:
        if (
            isinstance(expr, Field)
            and expr.offset == 0
            and isinstance(expr.base, Var)
            and expr.base.name in names
        ):
            return Field(expr.base, 0, "length", evidence=expr.evidence, rules=expr.rules)
        return expr

    for stmt in iter_statements(statements):
        _map_statement(stmt, fn)


def _assignments(statements: list[Stmt]) -> Counter[str]:
    return Counter(
        target.name
        for stmt in iter_statements(statements)
        if isinstance(stmt, (Assign, ParallelAssign))
        for target in stmt.targets
        if is_local(target)
    )


def memory_objects(state: FunctionState, public: bool) -> dict[str, ObjectKind]:
    """Allocations as values, words as fields; returns each object's kind."""
    assignments = _assignments(state.body)
    objects: dict[str, _Object] = {}
    scopes: dict[str, list[Stmt]] = {}
    state.body = _find(state.body, Tracker(frozenset()), assignments, objects, scopes, public)
    decoded = _decoded_objects(state.body, assignments)
    if decoded:
        rename_objects(state.body, decoded)
    if not objects:
        return decoded
    _rename(state.body, objects, {})
    for name, obj in objects.items():
        if obj.constant_base is None:
            continue
        offsets = range(0, obj.size, 32) if obj.size is not None else range(1)
        constants = {obj.constant_base + offset: (obj, offset) for offset in offsets}
        _rename(scopes[name], {}, constants)
    state.body = _literals(state.body, objects)
    constants = {name for name, obj in objects.items() if obj.kind == "bytes-literal"}
    if constants:
        _relabel_lengths(state.body, constants)
    lengths = {
        name: obj.statement.value.length if isinstance(obj.statement.value, Allocation) else None
        for name, obj in objects.items()
        if obj.kind in {"bytes", "bytes-literal"}
    }
    if lengths:
        bases = {name: objects[name].constant_base for name in lengths}
        _hashed_objects(state.body, lengths, bases)
    return decoded | {name: _kind(obj) for name, obj in objects.items()}


def _decoded_objects(statements: list[Stmt], assignments: Counter[str]) -> dict[str, ObjectKind]:
    """Results of abi.decode(returndata, (T)) for T bytes or an array (returndata.py)."""
    kinds: dict[str, ObjectKind] = {
        "bytes": ("bytes", None),
        "uint256[]": ("array", None),
        "bytes[]": ("array", None),
    }
    found: dict[str, ObjectKind] = {}
    for stmt in iter_statements(statements):
        if (
            isinstance(stmt, Assign)
            and isinstance(stmt.value, AbiCodec)
            and stmt.value.op == "decode"
            and dict(stmt.value.layout).get("solc") == "returndata-decoder"
        ):
            for target, kind in zip(stmt.targets, stmt.value.types, strict=False):
                if (
                    isinstance(target, Var)
                    and is_local(target)
                    and assignments[target.name] == 1
                    and kind in kinds
                ):
                    found[target.name] = kinds[kind]
    return found


def _hashed_objects(
    statements: list[Stmt],
    lengths: dict[str, Expr | None],
    bases: dict[str, int | None] | None = None,
) -> None:
    """keccak256(MEM[p + 32:p + 32 + n]) -> keccak256(p) for a bytes object p
    whose length is n: p.length, or the length the allocation wrote (the
    interpreter checks that it is still the length word when executed). For an
    object at a constant address A, MEM[A + 32:A + 32 + p.length] (p.length is
    printed only where the allocation dominates)."""

    def atoms(expr: Expr) -> Expr:
        def fn(node: Expr) -> Expr:
            if isinstance(node, Field) and node.offset == 0 and isinstance(node.base, Var):
                return Var(f"{node.base.name}.length")
            return node

        return map_expr(expr, fn)

    def fn(expr: Expr) -> Expr:
        if not (
            isinstance(expr, Call)
            and expr.name == "keccak256"
            and len(expr.args) == 1
            and isinstance(expr.args[0], Slice)
            and expr.args[0].space == "MEM"
        ):
            return expr
        data = expr.args[0]
        t = Tracker(frozenset())
        start = t.canon(atoms(data.start))
        size = add(t.canon(atoms(data.end)), start, -1)
        for name, length in lengths.items():
            constant = (bases or {}).get(name)
            at_constant = constant is not None and start == ("const", constant + 32)
            if start != add(("var", name), ("const", 32)) and not at_constant:
                continue
            if size == t.canon(Var(f"{name}.length")):
                known = None
            elif length is not None and not at_constant and size == t.canon(atoms(length)):
                known = length
            else:
                continue
            value = MemoryBytes(Var(name, None), known, evidence=data.all_evidence())
            return Call("keccak256", (value,), evidence=expr.evidence, rules=expr.rules)
        return expr

    for stmt in iter_statements(statements):
        _map_statement(stmt, fn)


# (kind, size): ("struct", bytes) or ("bytes", None) / ("array", None).
ObjectKind = tuple[str, int | None]


def _kind(obj: _Object) -> ObjectKind:
    return ("bytes", None) if obj.kind == "bytes-literal" else (obj.kind, obj.size)


def object_type(kind: ObjectKind) -> str:
    """The printed type of a pointer to a memory object of this kind."""
    if kind[0] == "struct":
        assert kind[1] is not None
        return f"struct({kind[1] // 32}) memory"
    return "bytes memory" if kind[0] == "bytes" else "uint256[] memory"


@dataclass
class FlowFunction:
    """A function as seen by object_flow: `objects` are its locals known to
    hold pointers to memory objects (from memory_objects), updated in place."""

    name: str
    body: list[Stmt]
    params: list[str]
    private: bool
    objects: dict[str, ObjectKind]


def _private_returns(function: FlowFunction) -> dict[int, ObjectKind]:
    returns = [
        s
        for s in iter_statements(function.body)
        if isinstance(s, Return) and "tail-call" not in s.rules
    ]
    if not returns or any(s.raw is not None for s in returns):
        return {}
    if len({len(s.values) for s in returns}) != 1:
        return {}
    result = {}
    for position in range(len(returns[0].values)):
        kinds = {
            function.objects.get(value.name) if is_local(value) else None
            for value in (s.values[position] for s in returns)
        }
        if len(kinds) == 1 and None not in kinds:
            result[position] = kinds.pop()  # type: ignore[assignment]
    return result


def object_flow(
    functions: list[FlowFunction], closed: bool
) -> tuple[dict[str, dict[int, ObjectKind]], dict[str, dict[int, ObjectKind]]]:
    """Memory objects passed to and returned from private functions.

    A private function returns an object when every return statement returns,
    at that position, a local holding one kind of object; its callers' results
    there are then that object. A parameter is an object when every call of
    the function passes one kind of object there (only when `closed`: every
    call site is visible, no function failed and no callee is unresolved).
    Iterated to a fixpoint; words then print as fields, exactly as MEM reads.
    Returns (returned objects, parameter objects) by function name and position.
    """
    by_name = {f.name: f for f in functions}
    assignments = {f.name: _assignments(f.body) for f in functions}
    # Locals defined only as copies of other locals (merges): an object when
    # every copied local holds one kind of object (a self-copy keeps it).
    copies: dict[str, dict[str, list[str]]] = {}
    for function in functions:
        found: dict[str, list[str | None]] = {}
        for stmt in iter_statements(function.body):
            if isinstance(stmt, Assign) and len(stmt.targets) == 1:
                pairs = [(stmt.targets[0], stmt.value)]
            elif isinstance(stmt, ParallelAssign):
                pairs = list(zip(stmt.targets, stmt.values, strict=False))
            else:
                continue
            for target, value in pairs:
                if isinstance(target, Var) and is_local(target):
                    source = value.name if isinstance(value, Var) and is_local(value) else None
                    found.setdefault(target.name, []).append(source)
        copies[function.name] = {
            name: [v for v in sources if v is not None]
            for name, sources in found.items()
            if len(sources) == assignments[function.name][name] and None not in sources
        }
    calls: dict[str, list[tuple[FlowFunction, Call, tuple[Expr, ...]]]] = {}
    for function in functions:
        for stmt in iter_statements(function.body):
            targets = stmt.targets if isinstance(stmt, Assign) else ()
            for expr in stmt.expressions():
                for node in iter_expr(expr):
                    if isinstance(node, Call) and node.name in by_name:
                        direct = isinstance(stmt, Assign) and stmt.value is node
                        calls.setdefault(node.name, []).append(
                            (function, node, targets if direct else ())
                        )
    returned: dict[str, dict[int, ObjectKind]] = {}
    parameters: dict[str, dict[int, ObjectKind]] = {}
    changed = True
    while changed:
        changed = False

        def learn(function: FlowFunction, name: str, kind: ObjectKind) -> None:
            nonlocal changed
            if name not in function.objects:
                function.objects[name] = kind
                changed = True

        for function in functions:
            for name, values in copies[function.name].items():
                kinds = {function.objects.get(v) for v in values if v != name}
                kind = kinds.pop() if len(kinds) == 1 else None
                if kind is not None:
                    learn(function, name, kind)
        for function in functions:
            if not function.private:
                continue
            found = _private_returns(function)
            returned[function.name] = found
            for caller, _, targets in calls.get(function.name, []):
                for position, kind in found.items():
                    if position < len(targets):
                        target = targets[position]
                        if (
                            isinstance(target, Var)
                            and is_local(target)
                            and assignments[caller.name][target.name] == 1
                        ):
                            learn(caller, target.name, kind)
            if not closed:
                continue
            sites = calls.get(function.name, [])
            if not sites:
                continue
            for position, param in enumerate(function.params):
                if assignments[function.name][param]:
                    continue
                kinds = set()
                for caller, call, _ in sites:
                    arg = call.args[position] if len(call.args) == len(function.params) else None
                    kinds.add(
                        caller.objects.get(arg.name)
                        if isinstance(arg, Var) and is_local(arg)
                        else None
                    )
                if len(kinds) == 1 and None not in kinds:
                    kind = kinds.pop()
                    learn(function, param, kind)
                    parameters.setdefault(function.name, {})[position] = kind
    return returned, parameters


def rename_objects(statements: list[Stmt], objects: dict[str, ObjectKind]) -> None:
    """MEM[p + c] -> p.wordN / p.length for locals p known to hold objects."""
    _rename(
        statements,
        {
            name: _Object(name, kind, size, None, None)  # type: ignore[arg-type]
            for name, (kind, size) in objects.items()
        },
        {},
    )
    lengths = {name: None for name, (kind, _) in objects.items() if kind == "bytes"}
    if lengths:
        _hashed_objects(statements, lengths)
