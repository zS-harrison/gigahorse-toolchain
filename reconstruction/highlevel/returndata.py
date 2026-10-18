"""Dynamic values decoded from a call's returned data: `abi.decode(returndata, (T))`.

After a call returning `bytes`, `uint256[]` or `bytes[]`, solc copies the
returned data to the free pointer, moves the pointer past it and decodes it
with its memory decoder: offset and length checks, an allocation per array
and per `bytes`, copies and zero words. The statements differ by compiler
version and pipeline (inlined or through helpers, copy loops or MCOPY), so a
candidate region is not matched statement by statement: it is replaced only
when executing it gives exactly what executing the replacement gives (the
interpreter's decoder, `DecompiledMachine.decode_returndata`): the same
memory, free pointer, result and reverts, on valid encodings of several
shapes, with junk around them, and on malformed ones (offsets out of range or
above 2**64 - 1, lengths past the end or above 2**64 - 1, short data).
Rule returndata-decode.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import product

from ..validation import UnsupportedExecution
from .idioms import Recognition
from .ir import (
    AbiCodec,
    Assign,
    Binary,
    Call,
    Const,
    Emit,
    Env,
    Expr,
    ExprStmt,
    ExternalCallExpr,
    Index,
    Jump,
    Member,
    ParallelAssign,
    Require,
    Return,
    Stmt,
    Var,
    While,
    is_local,
)
from .passes import (
    is_free_pointer_read,
    iter_expr,
    iter_statements,
    map_bodies,
    read_expressions,
    statement_ids,
)

U64 = (1 << 64) - 1
MAX_REGION = 80
BUFFER = 0x100
INITIAL_POINTER = 0x80  # solc's free pointer before any allocation
KINDS = ("bytes[]", "uint256[]", "bytes")
MAX_VALUES = 3
# solc 0.8: Panic(0x41), a zero word after each bytes; older: INVALID, and the
# zero word only after a length that is not a multiple of 32.
_SOLC = ("solc", "returndata-decoder")
LAYOUTS = (
    (_SOLC,),
    (_SOLC, ("invalid", True)),
    (_SOLC, ("invalid", True), ("pad", "conditional")),
    (_SOLC, ("pad", "conditional")),
)


def _word(value: int) -> bytes:
    return (value % (1 << 256)).to_bytes(32, "big")


def _pad(data: bytes) -> bytes:
    return data + b"\0" * (-len(data) % 32)


def _bytes_tail(data: bytes) -> bytes:
    return _word(len(data)) + _pad(data)


def _tail(kind: str, value) -> bytes:
    """The ABI tail of one dynamic value (its length word first)."""
    if kind == "bytes":
        return _bytes_tail(value)
    if kind == "uint256[]":
        return _word(len(value)) + b"".join(_word(v) for v in value)
    heads, tails, offset = [], [], 32 * len(value)
    for element in value:
        heads.append(_word(offset))
        tail = _bytes_tail(element)
        tails.append(tail)
        offset += len(tail)
    return _word(len(value)) + b"".join(heads) + b"".join(tails)


def _encode(kinds: tuple[str, ...], values: list) -> tuple[bytes, list[int]]:
    """abi.encode of dynamic values; also each value's tail offset."""
    heads, tails, offsets, offset = [], [], [], 32 * len(kinds)
    for kind, value in zip(kinds, values, strict=True):
        heads.append(_word(offset))
        offsets.append(offset)
        tail = _tail(kind, value)
        tails.append(tail)
        offset += len(tail)
    return b"".join(heads) + b"".join(tails), offsets


def _set(data: bytes, offset: int, value: int) -> bytes:
    return data[:offset] + _word(value) + data[offset + 32 :]


VALUES: dict[str, list] = {
    "bytes": [b"", b"\x01", b"q" * 31, b"r" * 32, b"s" * 33, bytes(range(70))],
    "uint256[]": [[], [7], [1, 2**255, 3], list(range(9))],
    "bytes[]": [[], [b""], [b"ab", b"x" * 40], [b"q" * 32, b"", b"z" * 65]],
}


def samples(kinds: tuple[str, ...]) -> list[bytes]:
    """Returned data to execute a candidate on: valid encodings of the values,
    and malformed ones (for each value: its offset, its length, its elements)."""
    count = max(len(VALUES[k]) for k in kinds)
    valid = [
        _encode(kinds, [VALUES[k][(i + j) % len(VALUES[k])] for j, k in enumerate(kinds)])[0]
        for i in range(count)
    ]
    biggest = [VALUES[k][-1] for k in kinds]
    longest, offsets = _encode(kinds, biggest)
    heads = 32 * len(kinds)
    result = list(valid)
    result.append(longest + b"\x07" * 45)  # trailing data
    # A non-standard first offset, with a junk word before the tails.
    shifted = b"".join(_word(o + 32) for o in offsets) + _word(0xEE) + longest[heads:]
    result.append(shifted)
    result += [b"", b"\x00" * (heads - 1), longest[:-32]]  # short, truncated
    for position, (kind, offset) in enumerate(zip(kinds, offsets, strict=True)):
        result.append(_set(longest, 32 * position, U64 + 1))  # offset above 2**64 - 1
        result.append(_set(longest, 32 * position, len(longest)))  # offset past the end
        result.append(_set(longest, offset, U64 + 1))  # length above 2**64 - 1
        result.append(_set(longest, offset, len(longest)))  # length past the end
        if kind == "bytes[]":
            head = offset + 32  # the first element's offset word
            result.append(_set(longest, head, U64 + 1))
            result.append(_set(longest, head, len(longest)))
            element = head + int.from_bytes(longest[head : head + 32], "big")
            result.append(_set(longest, element, U64 + 1))
            result.append(_set(longest, element, len(longest)))
    return result


def _head_words(region: list[Stmt]) -> int:
    """The number of head words from the decoder's first size check
    (`require(int256(r) >= 32 * n)`); 1 when there is none."""
    for stmt in iter_statements(region):
        if isinstance(stmt, Require) and stmt.payload is None:
            condition = stmt.condition
            if (
                isinstance(condition, Binary)
                and condition.op == "s>="
                and isinstance(condition.right, Const)
                and condition.right.value % 32 == 0
                and 32 <= condition.right.value <= 32 * MAX_VALUES
            ):
                return condition.right.value // 32
            return 1
    return 1


def _start(statements: list[Stmt], index: int) -> tuple[int, Var | None] | None:
    """The region start for RETURNDATACOPY(q, 0, returndata.length) at `index`,
    and the buffer local when it was read before the region (via-IR's call
    output pointer); None: the region itself reads the free pointer into q."""
    stmt = statements[index]
    if not (
        isinstance(stmt, ExprStmt)
        and isinstance(stmt.expr, Call)
        and stmt.expr.name == "RETURNDATACOPY"
        and len(stmt.expr.args) == 3
    ):
        return None
    destination, source, size = stmt.expr.args
    if not (isinstance(destination, Var) and is_local(destination)):
        return None
    if getattr(source, "value", None) != 0:
        return None
    start = index
    previous = statements[index - 1] if index > 0 else None
    length_local = None
    if (
        isinstance(previous, Assign)
        and len(previous.targets) == 1
        and is_local(previous.targets[0])
        and _is_returndata_length(previous.value)
    ):
        length_local = previous.targets[0].name  # type: ignore[attr-defined]
        start = index - 1
        previous = statements[index - 2] if index > 1 else None
    if not (_is_returndata_length(size) or getattr(size, "name", None) == length_local):
        return None
    if (
        isinstance(previous, Assign)
        and len(previous.targets) == 1
        and isinstance(previous.targets[0], Var)
        and previous.targets[0].name == destination.name
        and is_free_pointer_read(previous.value)
    ):
        return start - 1, None
    return start, destination


def _is_returndata_length(expr: Expr) -> bool:
    return (
        isinstance(expr, Member)
        and expr.name == "length"
        and isinstance(expr.base, Env)
        and expr.base.name == "returndata"
    )


def _effect_free(region: list[Stmt], pure: Callable[[str], bool], returns: bool = False) -> bool:
    """Memory writes, copies, checks and calls of effect-free private functions
    only (and no return, unless `returns`: a private function's body)."""
    for stmt in iter_statements(region):
        if isinstance(stmt, Emit) or (isinstance(stmt, Return) and not returns):
            return False
        for expr in stmt.expressions():
            for node in iter_expr(expr):
                if isinstance(node, ExternalCallExpr):
                    return False
                if isinstance(node, Call) and node.name in {"CALLDATACOPY", "CODECOPY"}:
                    return False
                if isinstance(node, Call) and node.name.startswith("fn_") and not pure(node.name):
                    return False
        if isinstance(stmt, (Assign, ParallelAssign)):
            for target in stmt.targets:
                if isinstance(target, Var) and target.ssa is None:
                    return False
                if isinstance(target, Index) and getattr(target.base, "name", "") in {
                    "STORAGE",
                    "TSTORAGE",
                }:
                    return False
    # A break/continue outside the region's own loops would leave it.
    return not any(isinstance(s, Jump) for s in _outside_loops(region))


def _outside_loops(statements: list[Stmt]):
    for stmt in statements:
        yield stmt
        if not isinstance(stmt, While):
            for block in stmt.blocks():
                yield from _outside_loops(block)


def _assigned(region: list[Stmt]) -> list[str]:
    names: list[str] = []
    for stmt in iter_statements(region):
        if isinstance(stmt, (Assign, ParallelAssign)):
            for target in stmt.targets:
                if isinstance(target, Var) and is_local(target) and target.name not in names:
                    names.append(target.name)
    return names


def _reads(statements) -> set[str]:
    names: set[str] = set()
    for stmt in statements:
        for expr in read_expressions(stmt):
            for node in iter_expr(expr):
                if isinstance(node, Var) and is_local(node):
                    names.add(node.name)
    return names


class _Checker:
    def __init__(self, decompiled, base: int):
        from .interpreter import DecompiledMachine

        self.machine = DecompiledMachine(decompiled)
        self.base = base
        self.memory = bytearray((i * 29 + 11) % 256 for i in range(0x1400))
        self.memory[0x40:0x60] = base.to_bytes(32, "big")

    def run(self, statements: list[Stmt], data: bytes, buffer: Var | None):
        from .interpreter import ViewMismatch, _Frame, _Halt

        machine = self.machine
        machine.prepare_fragment(self.memory)
        machine.returndata = data
        frame = _Frame({buffer.name: self.base} if buffer is not None else {})
        try:
            machine.block(statements, frame)
        except _Halt as halt:
            return ("halt", halt.data, halt.reverted), {}
        except (UnsupportedExecution, ViewMismatch, KeyError, AssertionError):
            return None, {}
        memory = bytes(machine.memory).rstrip(b"\0")
        return ("ok", memory), dict(frame.values)


def _decodes(
    checker: _Checker,
    region: list[Stmt],
    buffer: Var | None,
    kinds: tuple[str, ...],
    layout: tuple,
    assigned: list[str],
) -> list[set[str]] | None:
    """For each decoded value, the region's locals holding it after every valid
    case; None when an execution differs from the decoder's."""
    targets = tuple(Var(f"__decoded{i}__", f"__decoded{i}__") for i in range(len(kinds)))
    args: tuple[Expr, ...] = (Env("returndata"),) + ((buffer,) if buffer else ())
    replacement = [Assign(targets, AbiCodec("decode", args, kinds, layout))]
    candidates = [set(assigned) for _ in kinds]
    decoded_any = False
    for data in samples(kinds):
        compiled, values = checker.run(region, data, buffer)
        expected, decoded = checker.run(replacement, data, buffer)
        if compiled is None or expected != compiled:
            return None
        if compiled[0] == "ok":
            decoded_any = True
            for i, target in enumerate(targets):
                result = decoded[target.name]
                candidates[i] = {n for n in candidates[i] if values.get(n) == result}
            if not all(candidates):
                return None
    return candidates if decoded_any else None


def _match(
    checkers: list[_Checker],
    region: list[Stmt],
    buffer: Var | None,
    assigned: list[str],
    escaped: set[str],
) -> tuple[tuple[str, ...], tuple, list[str], int] | None:
    """(types, decoder layout, result locals, cases) when the region decodes the
    returned data as those types. With the buffer at solc's initial free pointer
    0x80 (the analysis folded it to a constant), the layout records that
    assumption, which the interpreter checks."""
    for checker in checkers:
        assumption = (("assumed_buffer", checker.base),) if checker.base != BUFFER else ()
        for kinds in product(KINDS, repeat=_head_words(region)):
            for layout in LAYOUTS:
                layout = layout + assumption
                candidates = _decodes(checker, region, buffer, kinds, layout, assigned)
                if candidates is None:
                    continue
                # A result read after the region names it; else the first assigned.
                # Every local read after the region must be one of the results.
                names = [next(n for n in assigned if n in (c & escaped or c)) for c in candidates]
                if escaped - set(names) or len(set(names)) != len(names):
                    continue
                return kinds, layout, names, len(samples(kinds))
    return None


def recognize_returndata_decodes(decompiled) -> list[Recognition]:
    views = {view.name: view for view in decompiled.views}

    def pure(name: str, seen: frozenset[str] = frozenset()) -> bool:
        view = views.get(name)
        if view is None or name in seen:
            return False
        inner = frozenset(seen | {name})
        return _effect_free(view.body, lambda n: pure(n, inner), returns=True)

    checkers = [_Checker(decompiled, BUFFER), _Checker(decompiled, INITIAL_POINTER)]
    results: list[Recognition] = []
    for view in decompiled.views:
        everything = list(iter_statements(view.body))

        def visit(statements: list[Stmt], view=view, everything=everything) -> list[Stmt]:
            index = 0
            while index < len(statements):
                found = _start(statements, index)
                if found is None:
                    index += 1
                    continue
                start, buffer = found
                replaced = False
                for end in range(index + 3, min(len(statements), start + MAX_REGION) + 1):
                    region = statements[start:end]
                    if not _effect_free(region, pure):
                        break
                    inside = {id(s) for s in iter_statements(region)}
                    free = _reads(iter_statements(region)) - set(_assigned(region))
                    if free - ({buffer.name} if buffer is not None else set()):
                        continue
                    assigned = _assigned(region)
                    outside = _reads(s for s in everything if id(s) not in inside)
                    escaped = set(assigned) & outside
                    if len(escaped) > MAX_VALUES:
                        continue
                    matched = _match(checkers, region, buffer, assigned, escaped)
                    if matched is None:
                        continue
                    kinds, layout, names, cases = matched
                    variables = {
                        t.name: t
                        for s in iter_statements(region)
                        if isinstance(s, (Assign, ParallelAssign))
                        for t in s.targets
                        if isinstance(t, Var)
                    }
                    args: tuple[Expr, ...] = (Env("returndata"),) + (
                        (buffer,) if buffer is not None else ()
                    )
                    copy = statements[index]
                    where = f" at {buffer.name}" if buffer is not None else ""
                    assumed = dict(layout).get("assumed_buffer") is not None
                    node = Assign(
                        tuple(Var(n, variables[n].ssa) for n in names),
                        AbiCodec("decode", args, kinds, layout),
                        evidence=copy.evidence,
                        hidden=frozenset().union(*(statement_ids(s) for s in region)),
                        rules=frozenset(
                            {"returndata-decode"}
                            | ({"assumes-folded-free-pointer"} if assumed else set())
                        ),
                        note=f"returned data copied to memory{where}, decoded with solc's "
                        f"checks; checked by executing the compiled code on {cases} inputs",
                        origin=copy.origin,
                    )
                    statements[start:end] = [node]
                    results.append(
                        Recognition(view.name, "returndata-decode", cases, ",".join(kinds))
                    )
                    index = start + 1
                    replaced = True
                    break
                if not replaced:
                    index += 1
            return statements

        view.body = map_bodies(view.body, visit)
        view.state.body = view.body
    return results
