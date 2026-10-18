"""Vyper external calls whose results are read back from the output buffer
(VyperExternalFixture; memory_forwarding.returned_words).

After `require(returndata.length >= n)`, the call's constant output buffer
holds its first n / 32 returned words; reads of them print as the call's
results, `(v1, v1_1) = target.value(x)`. In Vyper, 0x40 is an ordinary word.
Executed against Py-EVM with the contract's own returns and with stubs
returning short, exact and longer data.
"""

import json
import re
from pathlib import Path

import pytest

from reconstruction_tests.decompiled_support import compiler_fixture

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures/reconstruction"
STEMS = ["vyper-external-v0212", "vyper-external-v0310"]


@pytest.mark.parametrize("stem", STEMS)
def test_returned_words_print_as_call_results(tmp_path, stem):
    _, decompiled = compiler_fixture(stem, tmp_path / stem)
    text = decompiled.text
    # Outside the dispatcher (Vyper 0.2 extracts the selector through memory).
    functions = text[: text.index("function __function_selector__")]
    code = [line for line in functions.splitlines() if not line.strip().startswith("//")]
    assert not [line for line in code if "MEM[" in line]
    assert re.search(r"\((v\d+), \1_1\) = \S+\.value\(varg1\);", text)
    assert re.search(r"\((v\d+), \1_1, \1_2\) = \S+\.pair\(varg1\);", text)
    assert re.search(r"return (v\d+)_1 \* 100 \+ \1_2;", text)


@pytest.mark.parametrize("stem", STEMS)
def test_returned_words_match_evm(tmp_path, monkeypatch, record_property, stem):
    pytest.importorskip("eth_tester", reason="install the validation extra")
    from eth_abi import encode

    from reconstruction_tests.test_decompiled_execution import (
        Harness,
        StubbedReference,
        returning,
        words,
    )
    from reconstruction_tests.test_weth_execution import deploy_runtime

    manifest = json.loads((FIXTURES / f"{stem}.json").read_text())
    _, decompiled = compiler_fixture(stem, tmp_path / stem)
    ids = manifest["method_identifiers"]
    runtime = bytes.fromhex((FIXTURES / f"{stem}.hex").read_text().strip())
    blobs = [b"", b"\x01" * 31, words(5), words(2**256 - 1), words(9, 4), words(3, 8, 1)[:95]]
    stubs = {0x20000 + i: returning(blob) for i, blob in enumerate(blobs)}
    reference = StubbedReference(stubs)
    deploy_runtime(reference, runtime, {})
    sender = reference.tester.get_accounts()[1]
    sizes = {address: len(code) for address, code in stubs.items()}
    own = int(reference.address, 16)
    sizes[own] = len(runtime)
    harness = Harness(decompiled, reference, {}, monkeypatch)
    for signature in (
        "combined(address,uint256)",
        "both(address,uint256)",
        "save(address,uint256)",
    ):
        for target in (own, *stubs):
            for x in (0, 7, 2**200):
                data = bytes.fromhex(ids[signature]) + encode(
                    ["address", "uint256"], [target.to_bytes(20, "big"), x]
                )
                harness.check(data, sender=sender, code_sizes=sizes)
    assert 10 < harness.reverts < harness.cases
    record_property(f"{stem}_cases", harness.cases)
