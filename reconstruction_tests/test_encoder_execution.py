"""ABI encodings of dynamic values, recognized from the compiled statements (EncoderFixture).

Calldata `string` and `bytes[]` arguments encoded into hashes, external calls
and events; hashes of calldata strings; memory strings passed on through
pre-Cancun copy loops; string.concat; returned words decoded at the free
pointer. The printed constructs run against Py-EVM, including `bytes[]`
elements whose offsets or lengths fail solc's checks during the encoding.
"""

import json
from pathlib import Path

import pytest

from reconstruction_tests.decompiled_support import compiler_fixture

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures/reconstruction"
MASK = (1 << 256) - 1


def _body(text: str, name: str) -> str:
    start = text.index(f"function {name}(")
    return text[start : text.index("\n    }\n", start)]


def test_dynamic_encodings_print_as_calls_hashes_and_events(tmp_path):
    _, decompiled = compiler_fixture("encoder-optimized", tmp_path / "encoder")
    text = decompiled.text
    ids = json.loads((FIXTURES / "encoder-optimized.json").read_text())["method_identifiers"]
    register = _body(text, "0x" + ids[next(k for k in ids if k.startswith("register("))])
    # The commitment: abi.encode with a bytes[] calldata argument, hashed.
    assert "keccak256(abi.encode(v" in register and ", varg5, varg6, varg7))" in register
    # A calldata string and a bytes[] passed on whole; the event's string data.
    assert ".0xc0d147f7(varg0, varg1, varg2, varg4, varg7)" in register
    assert "abi.encodeWithSelector(0x88b3e206, v" in register and ", varg5))" in register
    assert "keccak256(abi.encodePacked(varg0))" in register
    assert ", varg1, varg0, msg.value, v" in register  # emit (label, owner | name, ...)
    assert "CALLDATACOPY" not in register
    # makeCommitment returns the hash without storing the encoding's length.
    commitment = _body(text, "0x" + ids[next(k for k in ids if k.startswith("makeCommitment("))])
    assert "return keccak256(abi.encode(" in commitment
    # Private functions taking a string memory: copied by loops before Cancun.
    assert 'abi.encodePacked(varg0, hex"2e657468")' in text
    assert ".price(varg1, varg0, 3)" in text
    assert "UNCERTAIN: event data layout" not in text
    # Returned words are decoded where the call put them; the pointer moves past them.
    assert "MEM[64] = v" not in _body(text, "renew")


def _call(ids: dict, signature: str, types: list[str], values: list) -> bytes:
    from eth_abi import encode

    return bytes.fromhex(ids[signature]) + encode(types, values)


COMMIT = ["string", "address", "uint256", "bytes32", "address", "bytes[]", "bool", "uint16"]


def _commit_values(name: str, data: list[bytes], resolver: str) -> list:
    owner = "0x" + "11" * 20
    return [name, owner, 31536000, b"\x07" * 32, resolver, data, True, 3]


def _bad_element(data: bytes, word: int, value: int) -> bytes:
    """`data` with the 32-byte word at argument word index `word` replaced."""
    offset = 4 + 32 * word
    return data[:offset] + (value & MASK).to_bytes(32, "big") + data[offset + 32 :]


def test_dynamic_encodings_match_evm(tmp_path, monkeypatch, record_property):
    pytest.importorskip("eth_tester", reason="install the validation extra")
    from reconstruction_tests.test_decompiled_execution import Harness
    from reconstruction_tests.test_execution import EVMReference

    stem = "encoder-optimized"
    manifest = json.loads((FIXTURES / f"{stem}.json").read_text())
    _, decompiled = compiler_fixture(stem, tmp_path / stem)
    ids = manifest["method_identifiers"]
    reference = EVMReference(manifest)
    sender = reference.tester.get_accounts()[0]
    resolver = "0x" + "22" * 20
    harness = Harness(decompiled, reference, {}, monkeypatch)
    signature = {k.split("(")[0]: k for k in ids}

    names = ["", "a", "alice", "x" * 31, "y" * 32, "z" * 33, "long-name-" * 9]
    arrays = [[], [b""], [b"\x01\x02"], [b"q" * 32, b"", b"r" * 45], [b"s" * 64] * 3]
    for name in names:
        for data in arrays:
            values = _commit_values(name, data, resolver)
            harness.check(_call(ids, signature["makeCommitment"], COMMIT, values), sender=sender)
        for value in (0, 9):
            values = _commit_values(name, arrays[3], resolver)
            harness.check(
                _call(ids, signature["register"], COMMIT, values), sender=sender, value=value
            )
        harness.check(
            _call(ids, signature["register"], COMMIT, _commit_values(name, [], resolver)),
            sender=sender,
        )
        harness.check(
            _call(ids, signature["renew"], ["string", "uint256"], [name, 5]), sender=sender, value=2
        )
        harness.check(_call(ids, signature["claim"], ["string"], [name]), sender=sender)
        harness.check(
            _call(ids, signature["priceOf"], ["string", "uint256"], [name, 7]), sender=sender
        )
        harness.check(
            _call(ids, signature["rent"], ["string", "uint256"], [name, 7]), sender=sender
        )
        harness.check(_call(ids, signature["rentBoth"], ["string"], [name]), sender=sender)

    # bytes[] elements are checked while they are encoded: an offset past the data,
    # a length above 2**64 - 1, a length past the end, and a negative offset (which
    # the signed comparisons accept).
    good = _call(
        ids, signature["makeCommitment"], COMMIT, _commit_values("bob", arrays[3], resolver)
    )
    head = int.from_bytes(good[4 + 32 * 5 : 4 + 32 * 6], "big") // 32  # the array's word index
    first = head + 1  # its first element's offset word
    element = first + int.from_bytes(good[4 + 32 * first : 4 + 32 * first + 32], "big") // 32
    for word, value in (
        (first, len(good)),
        (first, MASK - 31),
        (element, 1 << 64),
        (element, len(good)),
    ):
        for selector in ("makeCommitment", "register"):
            data = bytes.fromhex(ids[signature[selector]]) + good[4:]
            harness.check(_bad_element(data, word, value), sender=sender)
    assert harness.reverts >= 4
    assert harness.calls > 80 and harness.logs > 20
    record_property("encoder_cases", harness.cases)


def _copy_loop_statements(destination_base: str, after: list | None = None):
    from reconstruction.highlevel.ir import Assign, Binary, Const, Env, Index, Var, While

    def mem(key):
        return Index(Env("MEM"), key)

    i, n = Var("i", "i"), Var("n", "n")
    d, s = Var(destination_base, destination_base), Var("src", "src")
    pointer = Assign((Var("p", "p"),), mem(Const(0x40)))
    loop = [
        Assign((i,), Const(0)),
        While(
            [
                Assign((mem(Binary("+", i, Binary("+", d, Const(32)))),), mem(Binary("+", i, s))),
                Assign((i,), Binary("+", i, Const(32))),
            ],
            Binary("<", i, n),
        ),
        Assign((mem(Binary("+", Binary("+", d, Const(32)), n)),), Const(0)),
    ]
    return [pointer, *loop, *(after or [])]


def test_copy_loop_becomes_mcopy_only_into_new_memory():
    """solc 0.8's pre-Cancun copy loop + zero word -> MCOPY + zero word, when the
    destination is relative to a free-pointer read (new memory) and the counter is
    not read afterwards."""
    from reconstruction.highlevel.abi_codec import copy_loops
    from reconstruction.highlevel.ir import ExprStmt, Return, Var
    from reconstruction.highlevel.printer import expression

    converted = copy_loops(_copy_loop_statements("p"), {"p"})
    assert isinstance(converted[1], ExprStmt)
    assert expression(converted[1].expr) == "MCOPY(p + 32, src, n)"
    assert "assumes-solc-memory-discipline" in converted[1].rules
    assert len(converted) == 3  # pointer read, MCOPY, zero word
    # Into memory not known to be new: the word loop stays.
    assert len(copy_loops(_copy_loop_statements("q"), {"p"})) == 4
    # The counter's final value is read afterwards: the loop stays.
    kept = _copy_loop_statements("p", [Return((Var("i", "i"),))])
    assert len(copy_loops(kept, {"p"})) == 5
