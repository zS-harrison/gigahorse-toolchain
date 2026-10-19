"""Structuring shared regions, callee copies and msg.sig switches (Balancer, 1inch)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from reconstruction.controlflow import structure_function
from reconstruction.highlevel.view import classify_unprinted
from reconstruction.model import Block, Function, Instruction, Program

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures/reconstruction"


def _stmt(ident, opcode, definitions=(), operands=()):
    return Instruction(ident, opcode, list(definitions), list(operands), [ident])


def test_shared_region_with_several_returns_is_copied_per_caller():
    """An internal function inlined at two call sites that returns from two places:
    both return jumps go through the same return-address PHI (Balancer fn_0x23f4)."""
    blocks = [
        Block("0x0", [_stmt("0x1", "JUMPI", (), ("to_a", "c0"))], ["0x10", "0x20"], ["0x20"]),
        Block("0x10", [_stmt("0x11", "CONST", ("r1",)), _stmt("0x12", "JUMP")], ["0x50"], []),
        Block("0x20", [_stmt("0x21", "CONST", ("r2",)), _stmt("0x22", "JUMP")], ["0x50"], []),
        Block(
            "0x50",
            [
                _stmt("0x51", "PHI", ("ret",), ("r1", "r2")),
                _stmt("0x52", "JUMPI", (), ("to_x", "c1")),
            ],
            ["0x60", "0x70"],
            ["0x70"],
        ),
        Block("0x60", [_stmt("0x61", "JUMP", (), ("ret",))], ["0x100", "0x200"], []),
        Block("0x70", [_stmt("0x71", "JUMP", (), ("ret",))], ["0x100", "0x200"], []),
        Block("0x100", [_stmt("0x101", "STOP")], [], []),
        Block("0x200", [_stmt("0x201", "STOP")], [], []),
    ]
    function = Function("fn", ["0x0"], [], None, [], [b.id for b in blocks])
    program = Program(
        {"r1": "0x100", "r2": "0x200"},
        {"to_a": "0x10", "to_x": "0x60"},
        [function],
        blocks,
        [],
        [],
        phi_inputs=[("0x51", "0x10", "r1"), ("0x51", "0x20", "r2")],
    )
    plan = structure_function(program, function, split_shared_returns=True)
    assert plan.region is not None, plan.warnings
    # Each caller has its own copy of the region; both of its returns go back to it.
    returns = {}
    for node, transfer in plan.transfers.items():
        if plan.origin.get(node) in {"0x60", "0x70"}:
            returns.setdefault(node.split("@")[1], set()).update(transfer.targets)
    assert sorted(returns.values(), key=sorted) == [{"0x100"}, {"0x200"}]
    entries = {
        plan.transfers[caller].targets[0].split("@")[1]: caller for caller in ("0x10", "0x20")
    }
    assert {caller: returns[copy] for copy, caller in entries.items()} == {
        "0x10": {"0x100"},
        "0x20": {"0x200"},
    }


def test_call_with_two_continuations_is_split_by_its_return_address():
    """Two paths reach one call block, each having pushed its own return address
    (a PHI of constants there); the native analysis merged them, so the call has
    two continuations (1inch's call to 0x3e31). Each caller gets its own copy of
    the call block, continuing where its constant points. Block ids are clones
    of function 0x5 ('0x1d1b0x5' is address 0x1d1b in it)."""
    blocks = [
        Block(
            "0x00x5", [_stmt("0x1", "JUMPI", (), ("to_b", "c0"))], ["0xa0x5", "0xb0x5"], ["0xb0x5"]
        ),
        Block("0xa0x5", [_stmt("0xa1", "CONST", ("r1",)), _stmt("0xa2", "JUMP")], ["0x500x5"], []),
        Block("0xb0x5", [_stmt("0xb1", "CONST", ("r2",)), _stmt("0xb2", "JUMP")], ["0x500x5"], []),
        Block(
            "0x500x5",
            [
                _stmt("0x51", "PHI", ("ret",), ("r1", "r2")),
                _stmt("0x52", "CONST", ("callee",)),
                _stmt("0x53", "CALLPRIVATE", ("out",), ("callee", "arg")),
            ],
            ["0x1d1b0x5", "0x1db00x5"],
            [],
        ),
        Block("0x1d1b0x5", [_stmt("0x1d1c", "STOP")], [], []),
        Block("0x1db00x5", [_stmt("0x1db1", "STOP")], [], []),
        Block("0x90", [_stmt("0x91", "RETURNPRIVATE", (), ("out",))], [], []),
    ]
    caller = Function("0x5", ["0x00x5"], [], None, [], [b.id for b in blocks[:-1]])
    callee = Function("0x90", ["0x90"], [], None, ["arg"], ["0x90"])
    program = Program(
        {"r1": "0x1d1b", "r2": "0x1db0", "callee": "0x90"},
        {"to_b": "0xa0x5"},
        [caller, callee],
        blocks,
        [],
        [],
        phi_inputs=[("0x51", "0xa0x5", "r1"), ("0x51", "0xb0x5", "r2")],
    )
    returns = [("0x500x5", "0x90", "0x1d1b0x5"), ("0x500x5", "0x90", "0x1db00x5")]
    plan = structure_function(program, caller, call_returns=returns, split_shared_returns=True)
    assert plan.region is not None, plan.warnings
    continues = {
        source: plan.transfers[plan.transfers[source].targets[0]].targets
        for source in ("0xa0x5", "0xb0x5")
    }
    assert continues == {"0xa0x5": ("0x1d1b0x5",), "0xb0x5": ("0x1db00x5",)}
    # Without the PHI naming one continuation per edge, the CFG stays explicit.
    program.phi_inputs = [("0x51", "0xa0x5", "r1"), ("0x51", "0xb0x5", "r1")]
    plan = structure_function(program, caller, call_returns=returns, split_shared_returns=True)
    assert plan.region is None and "Ambiguous internal call continuation" in plan.warnings[0]


def _call_program(extra):
    """A call whose block also has an edge into `extra` (1inch fn_0x14b2's shape)."""
    blocks = [
        Block(
            "0x10",
            [
                _stmt("0x11", "CONST", ("callee",)),
                _stmt("0x12", "CALLPRIVATE", ("out",), ("callee", "arg")),
            ],
            ["0x20", extra],
            [],
        ),
        Block("0x20", [_stmt("0x21", "STOP")], [], []),
        Block(extra, [_stmt(f"{extra}s", "JUMP", (), ("unknown",))], [], []),
        Block("0x90", [_stmt("0x91", "RETURNPRIVATE", (), ("out",))], [], []),
    ]
    caller = Function("0x10", ["0x10"], [], None, [], ["0x10", "0x20", extra])
    callee = Function("0x90", ["0x90"], [], None, ["arg"], ["0x90"])
    program = Program({"callee": "0x90"}, {}, [caller, callee], blocks, [], [])
    return program, caller


def test_copy_of_the_callee_beside_a_call_is_not_executed():
    program, caller = _call_program("0x900x10")  # the callee's entry, copied into 0x10
    plan = structure_function(
        program, caller, call_returns=[("0x10", "0x90", "0x20")], split_shared_returns=True
    )
    assert plan.region is not None, plan.warnings
    call = plan.transfers["0x10"]
    assert call.targets == ("0x20",) and call.ignored_callee_copies == ("0x900x10",)
    assert "0x900x10" not in plan.reachable_blocks
    # Its statements are accounted for as a copy the call already runs.
    blocks = {b.id: b for b in program.blocks}
    context = SimpleNamespace(
        program=program,
        plans={"0x10": plan},
        blocks=blocks,
        statements={s.id: s for b in program.blocks for s in b.instructions},
        uses={},
    )
    assert classify_unprinted(context, set())["callee_copy_not_executed"] == {"0x900x10s"}
    # Any other second target (or a second continuation) still leaves the CFG explicit.
    program, caller = _call_program("0x30")
    plan = structure_function(program, caller, call_returns=[("0x10", "0x90", "0x20")])
    assert plan.region is None and "Ambiguous internal call continuation" in plan.warnings[0]
    program, caller = _call_program("0x900x10")
    plan = structure_function(
        program, caller, call_returns=[("0x10", "0x90", "0x20"), ("0x10", "0x90", "0x900x10")]
    )
    assert plan.region is None


@pytest.mark.parametrize("variant", ["optimized", "via-ir"])
def test_msg_sig_switch_in_an_internal_function_is_not_dispatch(variant, tmp_path, monkeypatch):
    pytest.importorskip("eth_tester", reason="install the validation extra")
    from eth_abi import encode

    from reconstruction_tests.decompiled_support import compiler_fixture
    from reconstruction_tests.test_decompiled_execution import Harness
    from reconstruction_tests.test_execution import EVMReference

    manifest = json.loads((FIXTURES / f"switch-{variant}.json").read_text())
    program, decompiled = compiler_fixture(f"switch-{variant}", tmp_path / variant)
    ids = manifest["method_identifiers"]
    # The switch compares first's and second's selectors again (EQ, then JUMPI); only
    # the dispatcher's comparisons start public functions (native rule).
    selectors = sorted(s for f in program.functions for s in f.selectors if s != "0x00000000")
    assert selectors == sorted(f"0x{i}" for i in ids.values())
    assert "not structured" not in decompiled.text
    reference = EVMReference(manifest)
    owner = reference.tester.get_accounts()[0]
    harness = Harness(decompiled, reference, {}, monkeypatch)
    for signature, values in (("first(uint256)", [7]), ("second(uint256)", [9]), ("other()", [])):
        types = ["uint256"] if values else []
        harness.check(bytes.fromhex(ids[signature]) + encode(types, values), sender=owner)
    harness.check(bytes.fromhex(ids["calls()"]), sender=owner)
    harness.check(bytes.fromhex(ids["other()"]), sender=owner)
    assert harness.cases == 5
