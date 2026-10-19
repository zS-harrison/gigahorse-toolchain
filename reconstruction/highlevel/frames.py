"""Vyper 0.2 frame saves and restores printed as one line each.

Around an internal call, Vyper 0.2 saves the caller's frame words (rewritten by
the native stack-loop pass into `v = MEM[a]` loads) and restores them after
(`MEM[a] = v`). A run of consecutive restores, each writing back the local that
was loaded from the same address, becomes `restoreFrame(MEM[a..b])`; a run of
consecutive loads whose locals are used only by such restores becomes
`saveFrame(MEM[a..b])`. The words are kept (`FrameWords`) and executed in
order, so behaviour is unchanged; only the printing groups them.
"""

from __future__ import annotations

from collections import Counter

from .ir import Assign, Const, Env, Expr, FrameWords, Index, Stmt, Var, is_local
from .passes import iter_expr, iter_statements, map_bodies, statement_ids

MIN_RUN = 3


def _load(stmt: Stmt) -> tuple[int, Var] | None:
    """`v = MEM[c]`."""
    if (
        isinstance(stmt, Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], Var)
        and is_local(stmt.targets[0])
        and isinstance(stmt.value, Index)
        and isinstance(stmt.value.base, Env)
        and stmt.value.base.name == "MEM"
        and isinstance(stmt.value.key, Const)
    ):
        return stmt.value.key.value, stmt.targets[0]
    return None


def _store(stmt: Stmt) -> tuple[int, Var] | None:
    """`MEM[c] = v`."""
    if (
        isinstance(stmt, Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], Index)
        and isinstance(stmt.targets[0].base, Env)
        and stmt.targets[0].base.name == "MEM"
        and isinstance(stmt.targets[0].key, Const)
        and isinstance(stmt.value, Var)
        and is_local(stmt.value)
    ):
        return stmt.targets[0].key.value, stmt.value
    return None


def group_frames(body: list[Stmt]) -> int:
    """Group frame restores, then saves, in a function body; returns lines saved."""
    assigned: Counter[str] = Counter()
    loaded_from: dict[str, int] = {}
    for stmt in iter_statements(body):
        for target in getattr(stmt, "targets", ()):
            if isinstance(target, Var) and is_local(target):
                assigned[target.name] += 1
        load = _load(stmt)
        if load is not None:
            loaded_from[load[1].name] = load[0]

    def restore(stmt: Stmt) -> tuple[int, Var] | None:
        # Writes back exactly what was loaded from that address.
        store = _store(stmt)
        if (
            store is not None
            and assigned[store[1].name] == 1
            and loaded_from.get(store[1].name) == store[0]
        ):
            return store
        return None

    saved = _group(body, "restore", restore)
    # Locals used only by grouped restores (and defined by a load) can be saved
    # in a group too: nothing printed names them individually.
    uses: Counter[str] = Counter()
    in_restores: Counter[str] = Counter()
    for stmt in iter_statements(body):
        if isinstance(stmt, FrameWords):
            for _, var in stmt.words:
                in_restores[var.name] += 1
            continue
        for expr in stmt.expressions():
            for node in iter_expr(expr):
                if isinstance(node, Var):
                    uses[node.name] += 1

    def save(stmt: Stmt) -> tuple[int, Var] | None:
        load = _load(stmt)
        if (
            load is not None
            and assigned[load[1].name] == 1
            and in_restores[load[1].name]
            and uses[load[1].name] == 1  # its own definition only
        ):
            return load
        return None

    saved += _group(body, "save", save)
    # Other runs of whole words at consecutive constant addresses, as tuples.
    saved += _group(body, "load", _load, consecutive=True)
    saved += _group(body, "store", _plain_store, consecutive=True)
    return saved


def _plain_store(stmt: Stmt) -> tuple[int, Expr] | None:
    """`MEM[c] = x` with x reading no memory and calling nothing."""
    from .ir import Call, ExternalCallExpr, Field, Slice

    if not (
        isinstance(stmt, Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], Index)
        and isinstance(stmt.targets[0].base, Env)
        and stmt.targets[0].base.name == "MEM"
        and isinstance(stmt.targets[0].key, Const)
        and stmt.targets[0].key.value != 0x40
    ):
        return None
    for node in iter_expr(stmt.value):
        if isinstance(node, (ExternalCallExpr, Slice, Field)):
            return None
        if (
            isinstance(node, Index)
            and isinstance(node.base, Env)
            and node.base.name
            in {
                "MEM",
                "MEM8",
            }
        ):
            return None
        if isinstance(node, Call) and not node.name.startswith(
            ("uint", "int", "bytes", "address", "bool")
        ):
            return None
    return stmt.targets[0].key.value, stmt.value


def _group(body: list[Stmt], kind: str, member, consecutive: bool = False) -> int:
    """Runs of at least MIN_RUN members at distinct addresses (consecutive words,
    ascending or descending, when `consecutive`)."""
    saved = 0

    def visit(statements: list[Stmt]) -> list[Stmt]:
        nonlocal saved
        out: list[Stmt] = []
        index = 0
        while index < len(statements):
            run: list[tuple[int, Var]] = []
            end = index
            while end < len(statements) and (word := member(statements[end])) is not None:
                if any(word[0] == other for other, _ in run):
                    break
                if consecutive and run:
                    step = word[0] - run[-1][0]
                    if abs(step) != 32 or (len(run) > 1 and step != run[1][0] - run[0][0]):
                        break
                run.append(word)
                end += 1
            if len(run) >= MIN_RUN:
                group = statements[index:end]
                out.append(
                    FrameWords(
                        kind,
                        tuple(run),
                        evidence=group[0].evidence,
                        hidden=frozenset().union(*(statement_ids(s) for s in group)),
                        rules=frozenset({"frame-words"}),
                        origin=group[0].origin,
                    )
                )
                saved += len(group) - 1
                index = end
                continue
            out.append(statements[index])
            index += 1
        return out

    body[:] = map_bodies(body, visit)
    return saved
