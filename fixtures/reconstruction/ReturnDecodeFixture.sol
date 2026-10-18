// SPDX-License-Identifier: MIT
pragma solidity 0.8.30;

/// Dynamic values decoded from a call's returned data: `bytes`, `uint256[]`,
/// `bytes[]` and a tuple of two arrays, each used after the call. The calls go
/// to the contract itself (or to a target given in calldata, so tests can
/// return malformed data); the callees only compute.
contract ReturnDecodeFixture {
    function blob(uint256 n) external pure returns (bytes memory out) {
        out = new bytes(n);
        for (uint256 i = 0; i < n; i++) {
            out[i] = bytes1(uint8(i * 7 + 1));
        }
    }

    function numbers(uint256 n) external pure returns (uint256[] memory out) {
        out = new uint256[](n);
        for (uint256 i = 0; i < n; i++) {
            out[i] = i * i + 3;
        }
    }

    function blobs(uint256 n) external pure returns (bytes[] memory out) {
        out = new bytes[](n);
        for (uint256 i = 0; i < n; i++) {
            out[i] = new bytes(i * 13);
        }
    }

    function pair(uint256 n) external pure returns (uint256[] memory a, uint256[] memory b) {
        a = new uint256[](n);
        b = new uint256[](n + 1);
        for (uint256 i = 0; i < n; i++) {
            a[i] = i + 10;
            b[i] = i + 20;
        }
    }

    function blobLength(address target, uint256 n) external view returns (uint256) {
        bytes memory data = ReturnDecodeFixture(target).blob(n);
        return data.length * 3 + (data.length > 0 ? uint8(data[0]) : 0);
    }

    function numberSum(address target, uint256 n) external view returns (uint256 total) {
        uint256[] memory values = ReturnDecodeFixture(target).numbers(n);
        for (uint256 i = 0; i < values.length; i++) {
            total += values[i];
        }
    }

    function blobsTotal(address target, uint256 n) external view returns (uint256 total) {
        bytes[] memory values = ReturnDecodeFixture(target).blobs(n);
        for (uint256 i = 0; i < values.length; i++) {
            total += values[i].length + 1;
        }
    }

    function pairLengths(address target, uint256 n) external view returns (uint256, uint256) {
        (uint256[] memory a, uint256[] memory b) = ReturnDecodeFixture(target).pair(n);
        return (a.length, b.length * 100 + (b.length > 0 ? b[0] : 0));
    }

    function echo(address target, uint256 n) external view returns (bytes memory) {
        return ReturnDecodeFixture(target).blob(n);
    }
}
