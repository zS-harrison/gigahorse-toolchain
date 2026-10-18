// SPDX-License-Identifier: MIT
pragma solidity 0.8.30;

/// Memory objects passed to and returned from private functions: a struct
/// returned by a helper (read as fields by its callers), `bytes` and `uint256[]`
/// parameters (lengths and elements), an array built and returned by a helper,
/// and hashes of a memory string's contents. Each helper has several callers
/// so the inliner keeps it a function.
contract ObjectFlowFixture {
    struct Price {
        uint256 base;
        uint256 premium;
    }

    mapping(bytes32 => uint256) public expiries;

    function _price(bytes memory name, uint256 duration) private view returns (Price memory p) {
        uint256 expiry = expiries[keccak256(name)];
        if (expiry > duration) {
            p = Price(duration * 3, expiry - duration);
        } else {
            p = Price(duration * 5 + name.length, 0);
        }
    }

    function _vowels(bytes memory name) private pure returns (uint256 count) {
        for (uint256 i = 0; i < name.length; i++) {
            bytes1 c = name[i];
            if (c == "a" || c == "e" || c == "i" || c == "o" || c == "u") {
                count++;
            }
        }
    }

    function _scaled(uint256[] memory values, uint256 factor) private pure returns (uint256[] memory out) {
        out = new uint256[](values.length);
        for (uint256 i = 0; i < values.length; i++) {
            out[i] = values[i] * factor + i;
        }
    }

    function _tail(uint256[] memory values, uint256 skip) private pure returns (uint256 total) {
        for (uint256 i = skip; i + 1 < values.length; i++) {
            total += values[i + 1] ^ values[i];
        }
    }

    function rentPrice(string calldata name, uint256 duration) external view returns (uint256, uint256) {
        Price memory p = _price(bytes(name), duration);
        return (p.base, p.premium);
    }

    function cost(string calldata name, uint256 duration) external view returns (uint256) {
        Price memory p = _price(bytes(name), duration);
        return p.base + p.premium + _vowels(bytes(name));
    }

    function register(string calldata name, uint256 duration) external payable {
        Price memory p = _price(bytes(name), duration);
        require(msg.value >= p.base + p.premium + _vowels(bytes(name)), "price");
        expiries[keccak256(bytes(name))] = duration;
    }

    function scaled(uint256[] calldata values, uint256 factor) external pure returns (uint256[] memory) {
        return _scaled(values, factor);
    }

    function mix(uint256[] calldata values, uint256 factor, uint256 skip) external pure returns (uint256) {
        return _tail(_scaled(values, factor), skip) + _tail(values, 0);
    }
}
