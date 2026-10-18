"""Memory objects across private calls (ObjectFlowFixture, objects.object_flow).

A struct returned by a helper is read as fields by its callers; `bytes` and
`uint256[]` parameters print lengths and elements; arrays built by a helper and
returned by a public function print `return v;`; hashes of a memory string
print `keccak256(v)`. The printed forms run against Py-EVM in both builds.
"""

import json
import re
from pathlib import Path

import pytest

from reconstruction.highlevel.ir import Assign, Binary, Call, Const, Env, Index, Return, Var
from reconstruction.highlevel.objects import FlowFunction, object_flow, rename_objects
from reconstruction.highlevel.printer import expression
from reconstruction_tests.decompiled_support import compiler_fixture

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures/reconstruction"
STEMS = ["object-flow-optimized", "object-flow-via-ir"]


def _body(text: str, header: str) -> str:
    start = text.index(header)
    return text[start : text.index("\n    }\n", start)]


@pytest.mark.parametrize("stem", STEMS)
def test_objects_print_across_private_calls(tmp_path, stem):
    _, decompiled = compiler_fixture(stem, tmp_path / stem)
    text = decompiled.text
    ids = json.loads((FIXTURES / f"{stem}.json").read_text())["method_identifiers"]
    # rentPrice returns the two words of the struct its helper returned.
    rent = _body(text, f"// 0x{ids['rentPrice(string,uint256)']}")
    assert re.search(r"return \((v\d+)\.word0, \1\.word1\);", rent), rent
    assert not re.search(r"MEM\[v\d+( \+ 32)?\]", rent), rent  # no raw struct words
    cost = _body(text, f"// 0x{ids['cost(string,uint256)']}")
    assert re.search(r"_SafeAdd\((v\d+)\.word0, \1\.word1\)", cost), cost
    # The array built by a helper is returned whole.
    scaled = _body(text, f"// 0x{ids['scaled(uint256[],uint256)']}")
    assert "returns (uint256[])" in scaled
    assert re.search(r"return v\d+; // returns the array of words", scaled), scaled
    # Elements at a variable index; no raw element words left in mix.
    mix = _body(text, f"// 0x{ids['mix(uint256[],uint256,uint256)']}")
    assert "MEM[" not in mix
    assert "uint256[] memory varg" in text
    assert re.search(r"\bv\w+\[v\d+( \+ 1)?\];", text)
    assert "(rule object-flow)" in text
    if stem == "object-flow-optimized":
        # Helpers take the objects; the struct helper's type is printed.
        assert re.search(
            r"function fn_0x\w+\(uint256 varg0, bytes memory varg1\) private "
            r"returns \(struct\(2\) memory\)",
            text,
        )
        assert re.search(r"function fn_0x\w+\(bytes memory varg0\) private", text)
        assert re.search(r"while \(v\d+ < varg0\.length\)", text)


def _call(ids: dict, signature: str, types: list[str], values: list) -> bytes:
    from eth_abi import encode

    return bytes.fromhex(ids[signature]) + encode(types, values)


@pytest.mark.parametrize("stem", STEMS)
def test_objects_across_private_calls_match_evm(tmp_path, monkeypatch, record_property, stem):
    pytest.importorskip("eth_tester", reason="install the validation extra")
    from reconstruction_tests.test_decompiled_execution import Harness
    from reconstruction_tests.test_execution import EVMReference

    manifest = json.loads((FIXTURES / f"{stem}.json").read_text())
    _, decompiled = compiler_fixture(stem, tmp_path / stem)
    ids = manifest["method_identifiers"]
    reference = EVMReference(manifest)
    sender = reference.tester.get_accounts()[0]
    harness = Harness(decompiled, reference, {}, monkeypatch)
    names = ["", "a", "aeiou", "bob", "x" * 31, "queue" * 7, "long-name-" * 9]
    for name in names:
        for duration in (0, 3, 1000):
            harness.check(
                _call(ids, "rentPrice(string,uint256)", ["string", "uint256"], [name, duration]),
                sender=sender,
            )
            harness.check(
                _call(ids, "cost(string,uint256)", ["string", "uint256"], [name, duration]),
                sender=sender,
            )
        # Too little value, then enough: the stored expiry changes the price.
        for value in (0, 10**6):
            harness.check(
                _call(ids, "register(string,uint256)", ["string", "uint256"], [name, 500]),
                sender=sender,
                value=value,
            )
        for duration in (3, 1000):
            harness.check(
                _call(ids, "rentPrice(string,uint256)", ["string", "uint256"], [name, duration]),
                sender=sender,
            )
    # Overflow in a helper reverts (Panic) in both.
    harness.check(
        _call(ids, "cost(string,uint256)", ["string", "uint256"], ["a", 2**255]), sender=sender
    )
    arrays = [[], [7], [1, 2], [5, 9, 13, 2], list(range(100, 109))]
    for values in arrays:
        for factor in (0, 3):
            harness.check(
                _call(ids, "scaled(uint256[],uint256)", ["uint256[]", "uint256"], [values, factor]),
                sender=sender,
            )
            for skip in (0, 1, 2, 20):
                harness.check(
                    _call(
                        ids,
                        "mix(uint256[],uint256,uint256)",
                        ["uint256[]", "uint256", "uint256"],
                        [values, factor, skip],
                    ),
                    sender=sender,
                )
    harness.check(
        _call(ids, "scaled(uint256[],uint256)", ["uint256[]", "uint256"], [[2**255], 2]),
        sender=sender,
    )
    assert harness.reverts >= 3
    record_property(f"{stem}_cases", harness.cases)


def _mem(key):
    return Index(Env("MEM"), key)


def _local(name):
    return Var(name, name)


def _functions(caller_arg_kind):
    """helper(p) returns p's word1; caller passes v (an object of caller_arg_kind)."""
    helper = FlowFunction(
        "helper",
        [Return((_mem(Binary("+", _local("p"), Const(32))),))],
        ["p"],
        True,
        {},
    )
    caller = FlowFunction(
        "caller",
        [Assign((_local("r"),), Call("helper", (_local("v"),)))],
        [],
        False,
        {"v": caller_arg_kind},
    )
    return helper, caller


def test_object_flow_parameters_need_every_call_site():
    helper, caller = _functions(("struct", 64))
    _, parameters = object_flow([helper, caller], closed=True)
    assert parameters == {"helper": {0: ("struct", 64)}}
    rename_objects(helper.body, {"p": ("struct", 64)})
    assert expression(helper.body[0].values[0]) == "p.word1"
    # A callee may be reached from code not seen (a failed view, an unresolved call).
    helper, caller = _functions(("struct", 64))
    assert object_flow([helper, caller], closed=False)[1] == {}
    # Calls passing different kinds, or a word: no claim.
    helper, caller = _functions(("bytes", None))
    other = FlowFunction(
        "other",
        [Assign((_local("s"),), Call("helper", (_local("w"),)))],
        [],
        False,
        {"w": ("struct", 64)},
    )
    assert object_flow([helper, caller, other], closed=True)[1] == {}


def test_object_flow_returns_need_every_return():
    returning = FlowFunction(
        "make",
        [Return((_local("a"),)), Return((_local("b"),))],
        [],
        True,
        {"a": ("struct", 64), "b": ("struct", 64)},
    )
    caller = FlowFunction("caller", [Assign((_local("r"),), Call("make", ()))], [], False, {})
    returned, _ = object_flow([returning, caller], closed=False)
    assert returned["make"] == {0: ("struct", 64)} and caller.objects == {"r": ("struct", 64)}
    returning.objects["b"] = ("struct", 96)
    caller.objects.clear()
    returned, _ = object_flow([returning, caller], closed=False)
    assert returned["make"] == {} and caller.objects == {}
