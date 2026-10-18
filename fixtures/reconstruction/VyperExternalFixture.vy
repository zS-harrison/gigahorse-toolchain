# External calls whose results Vyper reads back from the call's output buffer
# (one word, and a tuple of two), used in later arithmetic and a storage write.
# The calls go to a target given in calldata: the contract itself, or a stub.

interface Source:
    def value(x: uint256) -> uint256: view
    def pair(x: uint256) -> (uint256, uint256): view

stored: public(uint256)


@external
@view
def value(x: uint256) -> uint256:
    return x * 3 + 1


@external
@view
def pair(x: uint256) -> (uint256, uint256):
    return x + 7, x * 2


@external
@view
def combined(target: address, x: uint256) -> uint256:
    a: uint256 = Source(target).value(x)
    b: uint256 = Source(target).value(a)
    return a * 10 + b


@external
@view
def both(target: address, x: uint256) -> uint256:
    p: uint256 = 0
    q: uint256 = 0
    p, q = Source(target).pair(x)
    return p * 100 + q


@external
def save(target: address, x: uint256) -> uint256:
    v: uint256 = Source(target).value(x)
    self.stored = v + self.stored
    return self.stored
