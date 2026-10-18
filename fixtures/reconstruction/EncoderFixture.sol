// SPDX-License-Identifier: MIT
pragma solidity 0.8.30;

/// ABI encodings of dynamic values that solc builds statement by statement:
/// calldata `string` and `bytes[]` arguments encoded into hashes, external calls
/// and events, `bytes32` hashes of calldata strings, memory strings passed on
/// (compiled for Paris, where memory is copied with a word loop, not MCOPY), and
/// returned words decoded after the call moves the free pointer past them.
/// The external calls go to the contract itself, whose callees only compute.
contract EncoderFixture {
    event Registered(string name, bytes32 indexed label, address indexed owner, uint256 cost, uint256 expires);
    event Renewed(string name, bytes32 indexed label, uint256 cost, uint256 expires);

    mapping(bytes32 => uint256) public commitments;

    function makeCommitment(
        string memory name,
        address owner,
        uint256 duration,
        bytes32 secret,
        address resolver,
        bytes[] calldata data,
        bool reverse,
        uint16 fuses
    ) public pure returns (bytes32) {
        bytes32 label = keccak256(bytes(name));
        return keccak256(abi.encode(label, owner, duration, secret, resolver, data, reverse, fuses));
    }

    function register(
        string calldata name,
        address owner,
        uint256 duration,
        bytes32 secret,
        address resolver,
        bytes[] calldata data,
        bool reverse,
        uint16 fuses
    ) external payable {
        bytes32 commitment = makeCommitment(name, owner, duration, secret, resolver, data, reverse, fuses);
        commitments[commitment] = duration + 1;
        uint256 expires = EncoderFixture(address(this)).take(name, owner, duration, resolver, fuses);
        if (data.length > 0) {
            bytes32 node = keccak256(abi.encodePacked(bytes32(uint256(7)), keccak256(bytes(name))));
            EncoderFixture(address(this)).multi(node, data);
        }
        emit Registered(name, keccak256(bytes(name)), owner, msg.value, expires);
    }

    function renew(string calldata name, uint256 duration) external payable {
        bytes32 label = keccak256(bytes(name));
        (uint256 base, uint256 premium) = EncoderFixture(address(this)).price(name, duration, 3);
        uint256 expires = EncoderFixture(address(this)).take(name, msg.sender, duration, address(0), 0);
        emit Renewed(name, label, base + premium, expires);
    }

    function priceOf(string memory name, uint256 duration) public view returns (uint256, uint256) {
        return EncoderFixture(address(this)).price(name, duration, 3);
    }

    function claim(string memory name) public view returns (uint256) {
        return EncoderFixture(address(this)).take(string.concat(name, ".eth"), msg.sender, 1, address(0), 0);
    }

    function rent(string calldata name, uint256 duration) external view returns (uint256) {
        bytes32 label = keccak256(bytes(name));
        return _rent(name, duration) ^ uint256(label);
    }

    function rentBoth(string calldata name) external view returns (uint256) {
        return _rent(name, 1) + _rent(name, 2) + _claim(name) + _claim(name);
    }

    function _rent(string memory name, uint256 duration) internal view returns (uint256) {
        (uint256 base, uint256 premium) = EncoderFixture(address(this)).price(name, duration, 3);
        return base + premium;
    }

    function _claim(string memory name) internal view returns (uint256) {
        return EncoderFixture(address(this)).take(string.concat(name, ".eth"), msg.sender, 2, address(0), 0);
    }

    function take(string calldata name, address owner, uint256 duration, address resolver, uint16 fuses)
        external
        pure
        returns (uint256)
    {
        return bytes(name).length + (uint256(uint160(owner)) ^ uint256(uint160(resolver))) + duration + fuses;
    }

    function multi(bytes32 node, bytes[] calldata data) external pure returns (bytes[] memory results) {
        results = new bytes[](data.length);
        for (uint256 i = 0; i < data.length; i++) {
            results[i] = abi.encodePacked(node, data[i]);
        }
    }

    function price(string calldata name, uint256 duration, uint256 factor) external pure returns (uint256, uint256) {
        return (bytes(name).length * duration, factor);
    }
}
