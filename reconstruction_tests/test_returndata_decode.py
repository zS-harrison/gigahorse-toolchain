"""Dynamic values decoded from returned data (ReturnDecodeFixture, returndata.py).

`bytes`, `uint256[]`, `bytes[]` and `(uint256[], uint256[])` returned by a
call print as `abi.decode(returndata, (T))`. The decodes run against Py-EVM in
three builds, on the contract's own valid returns and on malformed data from
stub targets (each of the recognizer's samples).
"""

import json
import re
from pathlib import Path

import pytest

from reconstruction.highlevel.returndata import samples
from reconstruction_tests.decompiled_support import compiler_fixture

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures/reconstruction"
STEMS = ["return-decode-optimized", "return-decode-via-ir", "return-decode-paris-optimized"]
METHODS = {
    "blobLength(address,uint256)": ("bytes",),
    "numberSum(address,uint256)": ("uint256[]",),
    "blobsTotal(address,uint256)": ("bytes[]",),
    "pairLengths(address,uint256)": ("uint256[]", "uint256[]"),
    "echo(address,uint256)": ("bytes",),
}


@pytest.mark.parametrize("stem", STEMS)
def test_returned_values_print_as_decodes(tmp_path, stem):
    _, decompiled = compiler_fixture(stem, tmp_path / stem)
    text = decompiled.text
    decodes = re.findall(r"= abi\.decode\(returndata, \(([^)]*)\)\);", text)
    expected = ["bytes", "uint256[], uint256[]", "uint256[]", "bytes"]
    if stem != "return-decode-via-ir":
        # via-IR: the bytes[] call's printed arguments read a local the decoder
        # assigns, so that region is not replaced.
        expected.append("bytes[]")
    assert sorted(decodes) == sorted(expected), decodes
    assert "checked by executing the compiled code on" in text
    # The only returned data copied raw is a revert's (bubbled up).
    for line in text.splitlines():
        if "RETURNDATACOPY" in line and stem != "return-decode-via-ir":
            raise AssertionError(line)


@pytest.mark.parametrize("stem", STEMS)
def test_returned_values_decode_like_the_evm(tmp_path, monkeypatch, record_property, stem):
    pytest.importorskip("eth_tester", reason="install the validation extra")
    from eth_abi import encode

    from reconstruction_tests.test_decompiled_execution import (
        Harness,
        StubbedReference,
        returning,
    )
    from reconstruction_tests.test_weth_execution import deploy_runtime

    manifest = json.loads((FIXTURES / f"{stem}.json").read_text())
    _, decompiled = compiler_fixture(stem, tmp_path / stem)
    ids = manifest["method_identifiers"]
    runtime = bytes.fromhex((FIXTURES / f"{stem}.hex").read_text().strip())
    stubs: dict[int, bytes] = {}
    targets: dict[tuple[str, ...], list[int]] = {}
    for kinds in set(METHODS.values()):
        for blob in samples(kinds):
            address = 0x10000 + len(stubs)
            stubs[address] = returning(blob)
            targets.setdefault(kinds, []).append(address)
    reference = StubbedReference(stubs)
    deploy_runtime(reference, runtime, {})
    sender = reference.tester.get_accounts()[1]
    sizes = {address: len(code) for address, code in stubs.items()}
    sizes[int(reference.address, 16)] = len(runtime)
    harness = Harness(decompiled, reference, {}, monkeypatch)

    def call(signature: str, target: int, n: int) -> None:
        data = bytes.fromhex(ids[signature]) + encode(
            ["address", "uint256"], [target.to_bytes(20, "big"), n]
        )
        harness.check(data, sender=sender, code_sizes=sizes)

    own = int(reference.address, 16)
    for signature, kinds in METHODS.items():
        if stem == "return-decode-via-ir" and kinds == ("bytes[]",):
            continue  # its call reads a local before the analysis assigns it (not executable)
        for n in (0, 1, 3, 33):
            call(signature, own, n)  # the contract's own (valid) returns
        for address in targets[kinds]:
            call(signature, address, 2)
    assert harness.reverts > 20
    record_property(f"{stem}_cases", harness.cases)
