"""Execute the decompiled view itself.

The machine runs exactly the statement trees that `decompiled.sol.txt` prints.
Absorbed IR (hidden memory writes, elided reads) is *not* executed, so any
printed read that depended on a hidden write would produce a different result
than the EVM. Comparing this machine against Py-EVM therefore checks both the
expressions and the abstractions of the readable view on the tested inputs.

It is bounded validation, not a proof: unsupported constructs raise
UnsupportedExecution instead of being approximated.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ..keccak import keccak256
from ..validation import (
    ExecutionResult,
    ExternalInvocation,
    ExternalOutcome,
    RecoveredLog,
    UnsupportedExecution,
    signed,
)
from .ir import (
    AbiCodec,
    Allocation,
    Assign,
    Binary,
    Call,
    CalldataArg,
    Comment,
    Const,
    CustomError,
    Element,
    Emit,
    Env,
    Expr,
    ExprStmt,
    ExternalCallExpr,
    Field,
    FrameWords,
    Halt,
    If,
    Index,
    Jump,
    Member,
    MemoryBytes,
    ParallelAssign,
    Raw,
    RawBlock,
    Require,
    Return,
    Revert,
    Slice,
    Stmt,
    Unary,
    Var,
    While,
    is_local,
    struct_fields,
)
from .lift import DYNAMIC_MARKERS
from .parameters import member
from .view import DecompiledProgram, FunctionView

MASK = (1 << 256) - 1
ADDRESS = (1 << 160) - 1
U64 = (1 << 64) - 1
PANIC = bytes.fromhex("4e487b71")
# Printed environment names -> the opcode keys used by observed block values.
ENVIRONMENT_OPCODES = {
    "block.number": "NUMBER",
    "block.timestamp": "TIMESTAMP",
    "block.chainid": "CHAINID",
    "block.coinbase": "COINBASE",
    "block.basefee": "BASEFEE",
    "block.prevrandao": "PREVRANDAO",
    "block.gaslimit": "GASLIMIT",
    "block.blobbasefee": "BLOBBASEFEE",
    "tx.gasprice": "GASPRICE",
}


class ViewMismatch(AssertionError):
    """A printed claim contradicts what the printed statements execute."""


class _Halt(Exception):
    def __init__(self, data: bytes = b"", reverted: bool = False):
        self.data, self.reverted = data, reverted


class _Return(Exception):
    def __init__(self, values: list[int]):
        self.values = values


class _Loop(Exception):
    def __init__(self, kind: str):
        self.kind = kind


@dataclass
class _Frame:
    values: dict[str, int]


def _show(expr: Expr) -> str:
    from .printer import expression

    try:
        return expression(expr)
    except TypeError:
        return type(expr).__name__


def _cast(name: str, value: int) -> int:
    if name == "address":
        return value & ADDRESS
    if name == "bool":
        return int(value != 0)
    if name.startswith("uint") and name[4:].isdigit():
        return value & ((1 << int(name[4:])) - 1)
    if name.startswith("bytes") and name[5:].isdigit():
        size = int(name[5:])
        return value & (MASK ^ ((1 << (256 - 8 * size)) - 1))
    if name.startswith("int") and name[3:].isdigit():
        bits = int(name[3:])
        value &= (1 << bits) - 1
        if value >> (bits - 1):
            value |= MASK ^ ((1 << bits) - 1)
        return value
    raise UnsupportedExecution(f"Unknown cast {name}")


def _packed_width(kind: str) -> int:
    if kind == "address":
        return 20
    if kind.startswith("bytes") and kind[5:].isdigit():
        return int(kind[5:])
    if kind.startswith("uint") and kind[4:].isdigit():
        return int(kind[4:]) // 8
    raise UnsupportedExecution(f"Packed type {kind}")


def _member_path(expr: Expr) -> str | None:
    """`v0.f4.length` for a member chain over a local (values bound by abi.decode)."""
    names = []
    while isinstance(expr, Member):
        names.append(expr.name)
        expr = expr.base
    if not isinstance(expr, Var):
        return None
    return ".".join([expr.name, *reversed(names)])


def _signed(value: int) -> int:
    return value - (1 << 256) if value >> 255 else value


def _bytes_tail(payload: bytes) -> bytes:
    """A dynamic bytes value's ABI tail: length word, data, zero padding."""
    return len(payload).to_bytes(32, "big") + payload + b"\0" * (-len(payload) % 32)


def encode_error(text: str) -> bytes:
    data = text.encode()
    padded = data + b"\0" * (-len(data) % 32)
    return (
        bytes.fromhex("08c379a0")
        + (32).to_bytes(32, "big")
        + len(data).to_bytes(32, "big")
        + padded
    )


class DecompiledMachine:
    def __init__(
        self,
        decompiled: DecompiledProgram,
        *,
        external_handler: Callable[[ExternalInvocation], ExternalOutcome] | None = None,
        max_steps: int = 200000,
        max_memory: int = 1 << 20,
    ):
        self.program = decompiled
        self.functions = {view.name: view for view in decompiled.views}
        self.external_handler = external_handler
        self.max_steps, self.max_memory = max_steps, max_memory
        self.covered_lines: set[int] = set()
        self.trace: list[Stmt] = []

    # --- entry ---------------------------------------------------------------

    def execute(
        self,
        calldata: bytes,
        *,
        storage: dict[int, int] | None = None,
        caller: int = 1,
        address: int = 2,
        value: int = 0,
        observed_balance_values: Iterable[tuple[str, int, int]] = (),
        code_sizes: dict[int, int] | None = None,
        block_environment: dict[str, int] | None = None,
        block_hashes: dict[int, int] | None = None,
    ) -> ExecutionResult:
        original = dict(storage or {})
        self.storage = dict(original)
        self.memory = bytearray()
        self.calldata = calldata
        self.caller, self.address, self.callvalue = caller, address, value
        self.returndata = b""
        self.logs: list[RecoveredLog] = []
        self.external_calls: list[ExternalInvocation] = []
        self.steps = 0
        self.trace = []
        self.uncertain_executed = 0
        self.balances = iter(observed_balance_values)
        self.code_sizes = dict(code_sizes or {})
        self.block_environment = dict(block_environment or {})
        self.block_hashes = dict(block_hashes or {})
        dispatcher = self.functions.get("__function_selector__")
        if dispatcher is None:
            raise UnsupportedExecution("No selector dispatch function in the view")
        try:
            self.run_function(dispatcher, [])
        except _Halt as halt:
            return ExecutionResult(
                halt.data,
                halt.reverted,
                original if halt.reverted else dict(self.storage),
                [],
                self.steps,
                list(self.external_calls),
                [] if halt.reverted else list(self.logs),
            )
        raise UnsupportedExecution("Decompiled execution ended without a halt")

    def run_public(
        self,
        view: FunctionView,
        calldata: bytes,
        *,
        storage: dict[int, int] | None = None,
        caller: int = 1,
        address: int = 2,
        value: int = 0,
    ) -> ExecutionResult:
        """Run one public function body as if dispatched to it."""
        original = dict(storage or {})
        self.storage = dict(original)
        self.memory = bytearray(b"\0" * 0x40) + (0x80).to_bytes(32, "big")
        self.calldata = calldata
        self.caller, self.address, self.callvalue = caller, address, value
        self.returndata = b""
        self.logs, self.external_calls = [], []
        self.steps, self.trace, self.uncertain_executed = 0, [], 0
        self.balances = iter(())
        self.code_sizes, self.block_environment, self.block_hashes = {}, {}, {}
        try:
            self.run_function(view, [])
        except _Halt as halt:
            return ExecutionResult(
                halt.data,
                halt.reverted,
                original if halt.reverted else dict(self.storage),
                [],
                self.steps,
                list(self.external_calls),
                [] if halt.reverted else list(self.logs),
            )
        raise UnsupportedExecution("Function ended without a halt")

    def call_private(
        self, view: FunctionView, args: list[int], *, storage: dict[int, int] | None = None
    ) -> list[int] | None:
        """Run a private function on its own; None when it reverts."""
        self.storage = dict(storage or {})
        self.memory = bytearray(b"\0" * 0x40) + (0x80).to_bytes(32, "big")
        self.calldata, self.caller, self.address, self.callvalue = b"", 1, 2, 0
        self.returndata, self.logs, self.external_calls = b"", [], []
        self.steps, self.trace, self.uncertain_executed = 0, [], 0
        self.balances = iter(())
        self.code_sizes, self.block_environment, self.block_hashes = {}, {}, {}
        try:
            return self.run_function(view, args)
        except _Halt as halt:
            if halt.reverted:
                return None
            raise UnsupportedExecution("Private function halted the transaction") from None

    def prepare_fragment(self, memory: bytearray, storage: dict[int, int] | None = None) -> None:
        """Fresh state for executing a statement fragment on prepared memory."""
        self.storage = dict(storage or {})
        self.memory = bytearray(memory)
        self.calldata, self.caller, self.address, self.callvalue = b"", 1, 2, 0
        self.returndata, self.logs, self.external_calls = b"", [], []
        self.steps, self.trace, self.uncertain_executed = 0, [], 0
        self.balances = iter(())
        self.code_sizes, self.block_environment, self.block_hashes = {}, {}, {}

    def storage_string(self, root: Var, frame: _Frame) -> bytes:
        slot, _low, _high = self.storage_slot(root, frame)
        word = self.storage.get(slot, 0)
        if word & 1 == 0:
            length = (word & 0xFF) // 2
            return word.to_bytes(32, "big")[:length]
        length = (word - 1) // 2
        if length > self.max_memory:
            raise UnsupportedExecution("Stored string too long")
        start = int.from_bytes(keccak256(slot.to_bytes(32, "big")), "big")
        data = b"".join(
            self.storage.get(start + i, 0).to_bytes(32, "big") for i in range((length + 31) // 32)
        )
        return data[:length]

    def run_function(self, view: FunctionView, args: list[int]) -> list[int]:
        frame = _Frame({a.name: v for a, v in zip(view.arguments, args, strict=False)})
        if view.nonpayable is not None and self.callvalue:
            # The printed `nonpayable` stands for the compiler's leading callvalue check.
            raise _Halt(b"", True)
        if view.function.selectors:
            # Public arguments are ABI head words, enforced as the bytecode does:
            # masked at every use (old solc) or rejected when dirty (ABI validator).
            for index, argument in enumerate(view.arguments):
                if argument.definition is not None:
                    word = self.eval(argument.definition, frame)
                else:
                    word = self.calldata_word(4 + 32 * index)
                if argument.enforcement == "mask":
                    word = _cast(argument.type, word)
                elif argument.enforcement == "revert" and _cast(argument.type, word) != word:
                    raise _Halt(b"", True)
                frame.values[argument.name] = word
            # A static struct (array) parameter's members are its head words
            # (printed varg<i>.f<j>, resp. varg<i>[j]).
            for group, words, kind, _ in view.parameter_groups:
                for index, word_name in enumerate(words):
                    frame.values[f"{group}.{member(kind, index)}"] = frame.values[word_name]
        else:
            # A private parameter typed from masks: the callee cleans every use of it,
            # so the printed type stands for that cast (e.g. `address varg1`).
            for argument in view.arguments:
                if argument.enforcement == "mask" and argument.name in frame.values:
                    value = frame.values[argument.name]
                    frame.values[argument.name] = _cast(argument.type, value)
        try:
            self.block(view.body, frame)
        except _Return as result:
            return result.values
        if view.function.selectors or view.name == "__function_selector__":
            raise _Halt()
        return []

    # --- statements ----------------------------------------------------------

    def tick(self) -> None:
        self.steps += 1
        if self.steps > self.max_steps:
            raise UnsupportedExecution("Decompiled execution step limit exceeded")

    def block(self, statements: list[Stmt], frame: _Frame) -> None:
        for stmt in statements:
            self.statement(stmt, frame)

    def statement(self, stmt: Stmt, frame: _Frame) -> None:
        self.tick()
        self.trace.append(stmt)
        if len(self.trace) > 64:
            del self.trace[:32]
        if stmt.uncertain and not isinstance(stmt, Comment):
            # Executed anyway: a wrong printed meaning shows up as a divergence.
            self.uncertain_executed += 1
        if isinstance(stmt, Comment):
            return
        if isinstance(stmt, FrameWords):
            for address, value in stmt.words:
                if stmt.kind in {"save", "load"}:
                    assert isinstance(value, Var)
                    frame.values[value.name] = int.from_bytes(self.read(address, 32), "big")
                else:
                    self.write(address, self.eval(value, frame).to_bytes(32, "big"))
            return
        if isinstance(stmt, Assign):
            if isinstance(stmt.value, AbiCodec) and stmt.value.op == "decode":
                self.abi_decode(stmt.value, stmt.targets, frame)
                return
            if isinstance(stmt.value, AbiCodec) and len(stmt.targets) == 1:
                self.assign(stmt.targets[0], self.abi_encode(stmt.value, frame), frame)
                return
            if isinstance(stmt.value, Allocation) and len(stmt.targets) == 1:
                self.allocate_object(stmt.value, stmt.targets[0], frame)
                return
            if isinstance(stmt.value, ExternalCallExpr):
                results = self.external(stmt.value, frame, stmt)
                for target, value in zip(stmt.targets, results, strict=False):
                    self.assign(target, value, frame)
                return
            if isinstance(stmt.value, Call) and stmt.value.name in self.functions:
                results = self.private(stmt.value, frame)
                for target, value in zip(stmt.targets, results, strict=True):
                    self.assign(target, value, frame)
                return
            if len(stmt.targets) != 1:
                raise UnsupportedExecution("Multi-target assignment of a plain value")
            value = self.eval(stmt.value, frame)
            self.assign(stmt.targets[0], value, frame)
            return
        if isinstance(stmt, ParallelAssign):
            values = [self.eval(v, frame) for v in stmt.values]
            for target, value in zip(stmt.targets, values, strict=True):
                self.assign(target, value, frame)
            return
        if isinstance(stmt, ExprStmt):
            expr = stmt.expr
            if isinstance(expr, ExternalCallExpr):
                self.external(expr, frame, stmt)
            elif isinstance(expr, Call) and expr.name in self.functions:
                self.private(expr, frame)
            elif isinstance(expr, Call) and expr.name in {
                "CALLDATACOPY",
                "RETURNDATACOPY",
                "MCOPY",
                "CODECOPY",
            }:
                self.copy(expr, frame)
            else:
                self.eval(expr, frame)
            return
        if isinstance(stmt, Require):
            if self.truth(stmt.condition, frame):
                return
            if stmt.kind == "assert":
                raise _Halt(b"", True)
            raise _Halt(self.payload(stmt.payload, frame), True)
        if isinstance(stmt, If):
            branch = stmt.then if self.truth(stmt.condition, frame) else stmt.otherwise
            self.block(branch, frame)
            return
        if isinstance(stmt, While):
            while stmt.condition is None or self.truth(stmt.condition, frame):
                self.tick()
                try:
                    self.block(stmt.body, frame)
                except _Loop as action:
                    if action.kind == "break":
                        break
            return
        if isinstance(stmt, Jump):
            raise _Loop(stmt.kind)
        if isinstance(stmt, Return):
            if "return-memory-bytes" in stmt.rules:
                (pointer,) = stmt.values
                start = self.eval(pointer, frame)
                length = int.from_bytes(self.read(start, 32), "big")
                if length > self.max_memory:
                    raise UnsupportedExecution("Returned bytes too long")
                data = self.read(start + 32, length)
                padded = data + b"\0" * (-len(data) % 32)
                raise _Halt((32).to_bytes(32, "big") + length.to_bytes(32, "big") + padded)
            if "return-memory-array" in stmt.rules:
                (pointer,) = stmt.values
                start = self.eval(pointer, frame)
                length = int.from_bytes(self.read(start, 32), "big")
                if 32 * length > self.max_memory:
                    raise UnsupportedExecution("Returned array too long")
                data = self.read(start + 32, 32 * length)
                raise _Halt((32).to_bytes(32, "big") + length.to_bytes(32, "big") + data)
            if "storage-string-getter" in stmt.rules:
                (root,) = stmt.values
                assert isinstance(root, Var)
                data = self.storage_string(root, frame)
                padded = data + b"\0" * (-len(data) % 32)
                raise _Halt((32).to_bytes(32, "big") + len(data).to_bytes(32, "big") + padded)
            if "tail-call" in stmt.rules:
                (call,) = stmt.values
                assert isinstance(call, Call)
                self.private(call, frame)
                raise _Halt()
            if stmt.raw is not None:
                raise _Halt(self.slice(stmt.raw, frame))
            values = [self.eval(v, frame) for v in stmt.values]
            if "internal-return" in stmt.rules:
                raise _Return(values)
            raise _Halt(b"".join(v.to_bytes(32, "big") for v in values))
        if isinstance(stmt, Revert):
            raise _Halt(self.payload(stmt.payload, frame), True)
        if isinstance(stmt, Emit):
            self.emit(stmt, frame)
            return
        if isinstance(stmt, Halt):
            if stmt.kind == "stop":
                raise _Halt()
            if stmt.kind == "invalid":
                raise _Halt(b"", True)
            raise UnsupportedExecution("SELFDESTRUCT is not modelled")
        if isinstance(stmt, RawBlock):
            raise UnsupportedExecution("Unstructured control flow")
        raise UnsupportedExecution(f"Unsupported statement {type(stmt).__name__}")

    def assign(self, target: Expr, value: int, frame: _Frame) -> None:
        value &= MASK
        if is_local(target):
            frame.values[target.name] = value
            return
        if isinstance(target, Field):
            self.write(self.eval(target.base, frame) + target.offset, value.to_bytes(32, "big"))
            return
        if isinstance(target, Element):
            if target.space != "MEM":
                raise UnsupportedExecution("Assignment to calldata")
            self.write(self.element(target, frame), value.to_bytes(32, "big"))
            return
        if isinstance(target, Index) and isinstance(target.base, Env):
            if target.base.name == "MEM":
                self.write(self.eval(target.key, frame), value.to_bytes(32, "big"))
                return
            if target.base.name == "MEM8":
                self.write(self.eval(target.key, frame), bytes([value & 0xFF]))
                return
            if target.base.name == "STORAGE":
                self.storage[self.eval(target.key, frame)] = value
                return
        slot, low, high = self.storage_slot(target, frame)
        if (low, high) == (0, 31):
            self.storage[slot] = value
            return
        width = 8 * (high - low + 1)
        mask = ((1 << width) - 1) << (8 * low)
        old = self.storage.get(slot, 0)
        self.storage[slot] = (old & (MASK ^ mask)) | ((value << (8 * low)) & mask)

    # --- ABI coders (as solc compiles them) ------------------------------------

    def free_pointer(self) -> int:
        return int.from_bytes(self.read(0x40, 32), "big")

    def allocate_object(self, node: Allocation, target: Expr, frame: _Frame) -> None:
        """The statements an Allocation replaced, in their compiled order."""
        pointer = self.free_pointer()
        if node.assumed_base is not None and pointer != node.assumed_base:
            raise ViewMismatch(
                f"allocation: free pointer {pointer:#x}, expected {node.assumed_base:#x}"
            )
        self.assign(target, pointer, frame)
        self.write(0x40, self.eval(node.end, frame).to_bytes(32, "big"))
        if node.kind in {"bytes", "array"}:
            assert node.length is not None
            length = self.eval(node.length, frame)
            self.write(pointer, length.to_bytes(32, "big"))
            if node.copy is not None:
                source, size = (self.eval(e, frame) for e in node.copy)
                if size > self.max_memory:
                    raise UnsupportedExecution("Copy limit exceeded")
                if node.copy_space == "returndata":
                    if source + size > len(self.returndata):
                        raise _Halt(b"", True)
                    data = self.returndata[source : source + size]
                else:
                    data = self.calldata[source : source + size].ljust(size, b"\0")
                self.write(pointer + 32, data)
                if node.pad:
                    self.write(pointer + 32 + size, bytes(32))
        values = [self.eval(v, frame) for v in node.values]
        for index, value in enumerate(values):
            self.write(pointer + 32 * index, value.to_bytes(32, "big"))

    def allocate(self, pointer: int, size: int) -> None:
        """finalize_allocation: Panic(0x41) when the new pointer overflows 64 bits."""
        end = (pointer + size) & MASK
        if end > U64 or end < pointer:
            raise _Halt(PANIC + (0x41).to_bytes(32, "big"), True)
        self.write(0x40, end.to_bytes(32, "big"))

    def abi_decode(self, codec: AbiCodec, targets: tuple[Expr, ...], frame: _Frame) -> None:
        """abi.decode(data, (S)) of calldata bytes: solc 0.8's checked Yul decoder.

        Any malformed input reverts without data, except oversized lengths and
        allocations, which raise Panic(0x41), exactly in the decoder's order.
        """
        layout = dict(codec.layout)
        if layout.get("solc") == "returndata-decoder":
            self.decode_returndata(codec, targets, frame)
            return
        (source,) = codec.args
        (kind,) = codec.types
        struct = struct_fields(kind)
        if struct is None or len(targets) != 1 or not is_local(targets[0]):
            raise UnsupportedExecution(f"abi.decode to {kind}")
        fields = struct[1]
        target = targets[0].name
        position = self.eval(source, frame)
        start = (position + 32) & MASK
        end = (start + self.calldata_word(position)) & MASK
        if signed((end - start) & MASK) < 32:
            raise _Halt(b"", True)
        offset = self.calldata_word(start)
        if offset > U64:
            raise _Halt(b"", True)
        base = (start + offset) & MASK
        size = 32 * len(fields)
        if signed((end - base) & MASK) < size:
            raise _Halt(b"", True)
        pointer = self.free_pointer()
        expected = layout.get("struct_pointer")
        if expected is not None and pointer != expected:
            raise ViewMismatch(f"abi.decode: free pointer {pointer:#x}, expected {expected:#x}")
        self.allocate(pointer, size)
        frame.values[target] = pointer
        for index, field in enumerate(fields):
            head = (base + 32 * index) & MASK
            if field in {"bytes", "string"}:
                offset = self.calldata_word(head)
                if offset > U64:
                    raise _Halt(b"", True)
                tail = (base + offset) & MASK
                if not signed((tail + 31) & MASK) < signed(end):
                    raise _Halt(b"", True)
                length = self.calldata_word(tail)
                if length > U64:
                    raise _Halt(PANIC + (0x41).to_bytes(32, "big"), True)
                array = self.free_pointer()
                self.allocate(array, ((length + 31) & ~31) + 32)
                self.write(array, length.to_bytes(32, "big"))
                source_start = (tail + 32) & MASK
                if (source_start + length) & MASK > end:
                    raise _Halt(b"", True)
                if length > self.max_memory:
                    raise UnsupportedExecution("Decoded bytes too long")
                data = self.calldata[source_start : source_start + length].ljust(length, b"\0")
                self.write(array + 32, data)
                self.write(array + 32 + length, bytes(32))
                value = array
                frame.values[f"{target}.f{index}.length"] = length
            else:
                value = self.calldata_word(head)
                if field != "uint256" and _cast(field, value) != value:
                    raise _Halt(b"", True)
            self.write(pointer + 32 * index, value.to_bytes(32, "big"))
            frame.values[f"{target}.f{index}"] = value

    def decode_returndata(self, codec: AbiCodec, targets: tuple[Expr, ...], frame: _Frame) -> None:
        """abi.decode(returndata, (T)) as solc compiles it after a call: the returned
        data copied to the buffer (the free pointer, or `args[1]`), the pointer moved
        past it, then solc's memory decoder for T in `bytes`, `uint256[]`, `bytes[]`.

        Malformed data reverts without data; oversized lengths and allocations
        raise Panic(0x41) (`invalid` in the layout: INVALID, older solc), in the
        decoder's order. Each `bytes` is followed by a zero word (`pad`
        "conditional": only when its length is not a multiple of 32).
        """
        layout = dict(codec.layout)
        if len(targets) != len(codec.types) or not all(map(is_local, targets)):
            raise UnsupportedExecution("abi.decode(returndata): targets and types differ")
        failure = b"" if layout.get("invalid") else PANIC + (0x41).to_bytes(32, "big")
        # Older solc copies whole words and zeroes the word after the data only
        # when the length is not a multiple of 32.
        conditional_pad = layout.get("pad") == "conditional"
        data = self.returndata
        if len(data) > self.max_memory:
            raise UnsupportedExecution("Returned data too long")
        buffer = self.eval(codec.args[1], frame) if len(codec.args) > 1 else self.free_pointer()
        assumed = layout.get("assumed_buffer")
        if assumed is not None and buffer != assumed:
            raise ViewMismatch(f"abi.decode(returndata): buffer {buffer:#x}, assumed {assumed:#x}")

        def revert() -> None:
            raise _Halt(b"", True)

        def word(position: int) -> int:
            return int.from_bytes(self.read(position, 32), "big")

        def allocate(size: int) -> int:
            pointer = self.free_pointer()
            end = (pointer + ((size + 31) & ~31)) & MASK
            if end > U64 or end < pointer:
                raise _Halt(failure, True)
            self.write(0x40, end.to_bytes(32, "big"))
            return pointer

        def length_at(offset: int, end: int) -> int:
            if not signed((offset + 31) & MASK) < signed(end):
                revert()
            length = word(offset)
            if length > U64:
                raise _Halt(failure, True)
            return length

        def bytes_at(offset: int, end: int) -> int:
            length = length_at(offset, end)
            array = allocate(((length + 31) & ~31) + 32)
            self.write(array, length.to_bytes(32, "big"))
            source = (offset + 32) & MASK
            if (source + length) & MASK > end:
                revert()
            if length > self.max_memory:
                raise UnsupportedExecution("Decoded bytes too long")
            self.write(array + 32, self.read(source, length))
            if not conditional_pad or length % 32:
                self.write(array + 32 + length, bytes(32))
            return array

        def array_at(offset: int, end: int, element) -> int:
            length = length_at(offset, end)
            array = allocate(32 * length + 32)
            self.write(array, length.to_bytes(32, "big"))
            source = (offset + 32) & MASK
            if (source + 32 * length) & MASK > end:
                revert()
            for index in range(length):
                value = element(source, (source + 32 * index) & MASK, end)
                self.write(array + 32 + 32 * index, value.to_bytes(32, "big"))
            return array

        def word_element(_start: int, head: int, _end: int) -> int:
            return word(head)

        def bytes_element(start: int, head: int, end: int) -> int:
            offset = word(head)
            if offset > U64:
                revert()
            return bytes_at((start + offset) & MASK, end)

        self.write(buffer, data)
        end = (buffer + ((len(data) + 31) & ~31)) & MASK
        if end > U64 or end < buffer:
            raise _Halt(PANIC + (0x41).to_bytes(32, "big"), True)
        self.write(0x40, end.to_bytes(32, "big"))
        end = (buffer + len(data)) & MASK
        if signed((end - buffer) & MASK) < 32 * len(codec.types):
            revert()
        values = []
        for index, kind in enumerate(codec.types):
            offset = word(buffer + 32 * index)
            if offset > U64:
                revert()
            start = (buffer + offset) & MASK
            if kind == "bytes":
                values.append(bytes_at(start, end))
            elif kind == "uint256[]":
                values.append(array_at(start, end, word_element))
            elif kind == "bytes[]":
                values.append(array_at(start, end, bytes_element))
            else:
                raise UnsupportedExecution(f"abi.decode(returndata) to {kind}")
        for target, value in zip(targets, values, strict=True):
            self.assign(target, value, frame)

    def abi_encode(self, codec: AbiCodec, frame: _Frame) -> int:
        """A `bytes memory` encoding at the free pointer; returns its pointer."""
        layout = dict(codec.layout)
        if codec.op == "encode" and layout.get("solc") == "0.8-legacy-struct-literal":
            return self.encode_struct_literal(codec, layout, frame)
        if codec.op == "encodePacked" and layout.get("solc") == "legacy-packed-bytes":
            return self.encode_packed_bytes(codec, frame)
        raise UnsupportedExecution(f"abi.{codec.op} layout {layout.get('solc')}")

    def encode_struct_literal(self, codec: AbiCodec, layout: dict, frame: _Frame) -> int:
        """abi.encode(S(...)): allocate S (unchecked), copy its calldata bytes member to
        memory, then write offset word, heads, the member's length, data and a zero
        word, the length word and the new free pointer (solc 0.8 legacy codegen)."""
        (literal,) = codec.args
        (kind,) = codec.types
        struct = struct_fields(kind)
        if (
            struct is None
            or not isinstance(literal, Call)
            or layout.get("calldata_members") is None
        ):
            raise UnsupportedExecution(f"abi.encode of {kind}")
        fields = struct[1]
        (member,) = layout["calldata_members"]
        size = 32 * len(fields)
        pointer = self.free_pointer()
        self.write(0x40, ((pointer + size) & MASK).to_bytes(32, "big"))
        heads: list[int] = []
        copy = 0
        for index, (field, value) in enumerate(zip(fields, literal.args, strict=True)):
            if index != member:
                heads.append(self.eval(value, frame))
                continue
            if field != "bytes":
                raise UnsupportedExecution(f"abi.encode member {field}")
            position = self.eval(value, frame)
            length = self.calldata_word(position)
            if length > self.max_memory:
                raise UnsupportedExecution("Encoded bytes too long")
            data = self.calldata[position + 32 : position + 32 + length].ljust(length, b"\0")
            copy = self.free_pointer()
            self.write(0x40, ((copy + (length + 31) // 32 * 32 + 32) & MASK).to_bytes(32, "big"))
            self.write(copy, length.to_bytes(32, "big"))
            self.write(copy + 32, data)
            self.write(copy + 32 + length, bytes(32))
            heads.append(size)
        buffer = self.free_pointer()
        out = buffer + 32
        self.write(out, (32).to_bytes(32, "big"))
        for index, head in enumerate(heads):
            self.write(out + 32 + 32 * index, (head & MASK).to_bytes(32, "big"))
        length = int.from_bytes(self.read(copy, 32), "big")
        if length > self.max_memory:
            raise UnsupportedExecution("Encoded bytes too long")
        tail = out + 32 + size
        self.write(tail, length.to_bytes(32, "big"))
        self.write(tail + 32, self.read(copy + 32, length))
        self.write(tail + 32 + length, bytes(32))
        total = 32 + size + 32 + (length + 31) // 32 * 32
        self.write(buffer, total.to_bytes(32, "big"))
        self.write(0x40, ((out + total) & MASK).to_bytes(32, "big"))
        return buffer

    def encode_packed(self, codec: AbiCodec, frame: _Frame) -> bytes:
        """abi.encodePacked of static values (legacy codegen): each value is written as a
        left-aligned word at its offset after the length word, the length at the free
        pointer, and the pointer moves to the end of the data (not rounded)."""
        layout = dict(codec.layout)
        if layout.get("solc") == "legacy-packed-calldata":
            # One calldata bytes value: its bytes at the free pointer (not moved),
            # then a zero word.
            (value,) = codec.args
            assert isinstance(value, CalldataArg)
            data = self.calldata_bytes(self.eval(value.pointer, frame))
            pointer = self.free_pointer()
            self.write(pointer, data)
            self.write(pointer + len(data), bytes(32))
            return data
        if layout.get("solc") != "legacy-packed":
            raise UnsupportedExecution(f"abi.encodePacked layout {layout.get('solc')}")
        pointer = self.free_pointer()
        expected = layout.get("buffer_pointer")
        if expected is not None and pointer != expected:
            raise ViewMismatch(
                f"abi.encodePacked: free pointer {pointer:#x}, expected {expected:#x}"
            )
        data = b""
        for value, kind in zip(codec.args, codec.types, strict=True):
            width = _packed_width(kind)
            chunk = (self.eval(value, frame) & ((1 << (8 * width)) - 1)).to_bytes(width, "big")
            self.write(pointer + 32 + len(data), chunk.ljust(32, b"\0"))
            data += chunk
        self.write(pointer, len(data).to_bytes(32, "big"))
        self.write(0x40, ((pointer + 32 + len(data)) & MASK).to_bytes(32, "big"))
        return data

    def encode_packed_bytes(self, codec: AbiCodec, frame: _Frame) -> int:
        """abi.encodePacked with bytes elements (legacy codegen): each element after
        the length word at the free pointer (bytes followed by a zero word, a static
        value as a left-aligned word), then the length and the pointer moved to the
        end (not rounded). Returns the bytes value's pointer."""
        pointer = self.free_pointer()
        position = pointer + 32
        for value, kind in zip(codec.args, codec.types, strict=True):
            if kind == "bytes":
                if isinstance(value, MemoryBytes):
                    data = self.memory_bytes(value, frame)
                else:
                    assert isinstance(value, CalldataArg)
                    data = self.calldata_bytes(self.eval(value.pointer, frame))
                self.write(position, data)
                self.write(position + len(data), bytes(32))
                position += len(data)
                continue
            width = _packed_width(kind)
            chunk = (self.eval(value, frame) & ((1 << (8 * width)) - 1)).to_bytes(width, "big")
            self.write(position, chunk.ljust(32, b"\0"))
            position += width
        self.write(pointer, (position - pointer - 32).to_bytes(32, "big"))
        self.write(0x40, (position & MASK).to_bytes(32, "big"))
        return pointer

    def encode_tuple(self, codec: AbiCodec, frame: _Frame) -> bytes:
        """abi.encode(values) for a hash (legacy codegen): the encoding at the free
        pointer + 32, then (if compiled) its length at the pointer and the pointer
        moved past it."""
        layout = dict(codec.layout)
        if layout.get("solc") != "legacy-tuple":
            raise UnsupportedExecution(f"abi.encode layout {layout.get('solc')}")
        pointer = self.free_pointer()
        data = self.write_encoding(pointer + 32, codec.args, frame)
        if layout.get("length_word"):
            self.write(pointer, len(data).to_bytes(32, "big"))
            self.write(0x40, ((pointer + 32 + len(data)) & MASK).to_bytes(32, "big"))
        return data

    def write_encoding(self, start: int, args: Iterable[Expr], frame: _Frame) -> bytes:
        """Write the ABI encoding of `args` at `start` as solc does, and return it.

        solc writes a zero word after each tail's data; the last one reaches past
        the encoding and is written too.
        """
        data, trailing = self.encode_values(args, frame)
        self.write(start, data)
        if trailing is not None:
            self.write(start + trailing, bytes(32))
        return data

    def encode_values(self, args: Iterable[Expr], frame: _Frame) -> tuple[bytes, int | None]:
        """The ABI encoding of printed values, and where the zero word after the last
        copied tail starts. A dynamic value is `emptyBytes()`, a MemoryBytes or a
        CalldataArg; a `bytes[]` element failing solc's checks reverts without data."""
        args = list(args)
        heads: list[int] = []
        tails = b""
        trailing = None
        for arg in args:
            if isinstance(arg, Call) and arg.name == "emptyBytes":
                heads.append(32 * len(args) + len(tails))
                tails += bytes(32)
                trailing = None
            elif isinstance(arg, (MemoryBytes, CalldataArg)):
                heads.append(32 * len(args) + len(tails))
                encoded, inner = self.encode_dynamic(arg, frame)
                trailing = None if inner is None else 32 * len(args) + len(tails) + inner
                tails += encoded
            else:
                heads.append(self.eval(arg, frame))
        return b"".join((h & MASK).to_bytes(32, "big") for h in heads) + tails, trailing

    def encode_dynamic(self, value: Expr, frame: _Frame) -> tuple[bytes, int | None]:
        if isinstance(value, MemoryBytes):
            payload = self.memory_bytes(value, frame)
            return _bytes_tail(payload), 32 + len(payload)
        assert isinstance(value, CalldataArg)
        position = self.eval(value.pointer, frame)
        if value.abi_type == "bytes":
            payload = self.calldata_bytes(position)
            return _bytes_tail(payload), 32 + len(payload)
        if value.abi_type != "bytes[]":
            raise UnsupportedExecution(f"Calldata argument of type {value.abi_type}")
        count = self.calldata_word(position)
        if count > self.max_memory // 32:
            raise UnsupportedExecution("Encoded array too long")
        base = (position + 32) & MASK
        size = len(self.calldata)
        heads: list[int] = []
        tails = b""
        trailing = None
        for index in range(count):
            relative = self.calldata_word((base + 32 * index) & MASK)
            if not _signed(relative) < _signed((size - base - 31) & MASK):
                raise _Halt(b"", True)
            element = (base + relative) & MASK
            length = self.calldata_word(element)
            if length > (1 << 64) - 1:
                raise _Halt(b"", True)
            if not _signed((element + 32) & MASK) <= _signed((size - length) & MASK):
                raise _Halt(b"", True)
            payload = self.calldata_bytes(element)
            heads.append(32 * count + len(tails))
            trailing = 32 + 32 * count + len(tails) + 32 + len(payload)
            tails += _bytes_tail(payload)
        words = b"".join(h.to_bytes(32, "big") for h in heads)
        return count.to_bytes(32, "big") + words + tails, trailing

    def calldata_bytes(self, position: int) -> bytes:
        """The bytes of a calldata `bytes` value whose length word is at `position`."""
        length = self.calldata_word(position)
        if length > self.max_memory:
            raise UnsupportedExecution("Calldata bytes too long")
        start = position + 32
        return self.calldata[start : start + length].ljust(length, b"\0")

    def stage_input(self, selector: bytes, args: Iterable[Expr], frame: _Frame) -> bytes:
        """A call's input as solc writes it at the free pointer (not moved): the
        selector word, then the encoding after the selector."""
        start = self.free_pointer()
        self.write(start, selector.ljust(32, b"\0"))
        return selector + self.write_encoding(start + 4, args, frame)

    def stage_raw_input(self, stmt: Stmt, data: bytes) -> None:
        """Memory writes solc makes for a low-level call's input, where recognized."""
        buffer = stmt.meta.get("selector_buffer")
        if buffer is not None:
            # abi.encodeWithSelector: length and bytes at the pointer, which moves to
            # their end (not rounded); then staged at the new pointer for the call.
            pointer = self.free_pointer()
            self.write(pointer, len(data).to_bytes(32, "big"))
            self.write(pointer + 32, data)
            staging = pointer + 32 + len(data)
            self.write(0x40, (staging & MASK).to_bytes(32, "big"))
            if buffer["copy"]:
                self.write(staging, data)
            if buffer["zero"]:
                self.write(staging + len(data), bytes(32))
        if "bytes-staged-at-free-pointer" in stmt.rules:
            # target.call(b): b's data copied to the pointer, then a zero word.
            staging = self.free_pointer()
            self.write(staging, data)
            self.write(staging + len(data), bytes(32))

    def element(self, expr: Element, frame: _Frame) -> int:
        base, index = self.eval(expr.base, frame), self.eval(expr.index, frame)
        return (base + 32 + 32 * index) & MASK

    def memory_bytes(self, value: MemoryBytes, frame: _Frame) -> bytes:
        pointer = self.eval(value.pointer, frame)
        if value.length is not None:
            length = self.eval(value.length, frame)
        else:
            length = int.from_bytes(self.read(pointer, 32), "big")
        if length > self.max_memory:
            raise UnsupportedExecution("Memory bytes too long")
        return self.read(pointer + 32, length)

    # --- calls ---------------------------------------------------------------

    def private(self, call: Call, frame: _Frame) -> list[int]:
        view = self.functions[call.name]
        args = [self.eval(a, frame) for a in call.args]
        return self.run_function(view, args)

    def external(self, call: ExternalCallExpr, frame: _Frame, stmt: Stmt) -> list[int]:
        if self.external_handler is None:
            raise UnsupportedExecution("External calls need a replay handler")
        target = self.eval(call.target, frame) & ADDRESS
        value = self.eval(call.value, frame) if call.value is not None else 0
        if call.kind in {"eth-transfer", "eth-send"}:
            value = self.eval(call.args[0], frame)
            data = b""
        elif call.raw_input is not None and "staged-at-free-pointer" in stmt.rules:
            # abi.encodeWithSelector(selector, args) written at the free pointer.
            encoded = call.raw_input
            assert isinstance(encoded, Call) and isinstance(encoded.args[0], Raw)
            selector = bytes.fromhex(encoded.args[0].text.split()[0][2:])
            data = self.stage_input(selector, encoded.args[1:], frame)
        elif call.raw_input is not None:
            data = self.payload_bytes(call.raw_input, frame)
            self.stage_raw_input(stmt, data)
        elif stmt.meta.get("input") is not None and (
            stmt.uncertain
            or any(isinstance(a, Call) and a.name in DYNAMIC_MARKERS for a in call.args)
        ):
            # Arguments shown may not cover the input: send the printed memory bytes,
            # and require every printed argument to agree with them.
            start, end = (self.eval(e, frame) for e in stmt.meta["input"])
            data = self.read(start, (end - start) & MASK)
            self.check_arguments(call, stmt, data, start, frame)
        else:
            selector = self.selector_of(stmt)
            assert selector is not None
            if "staged-at-free-pointer" in stmt.rules:
                # solc writes the input at the free pointer (without moving it):
                # selector word, heads, then the tail (length, data and a zero word).
                data = self.stage_input(selector, call.args, frame)
            else:
                data = selector + self.encode_values(call.args, frame)[0]
        opcode = {
            "call": "CALL",
            "staticcall": "STATICCALL",
            "delegatecall": "DELEGATECALL",
            "callcode": "CALLCODE",
            "eth-transfer": "CALL",
            "eth-send": "CALL",
        }[call.kind]
        invocation = ExternalInvocation(stmt.origin or "", opcode, target, value, data, -1, {})
        self.external_calls.append(invocation)
        outcome = self.external_handler(invocation)
        self.returndata = outcome.data
        output = stmt.meta.get("output")
        if output is not None:
            start, size = self.eval(output[0], frame), self.eval(output[1], frame)
            if size:
                self.write(start, outcome.data[:size])
        if stmt.meta.get("returns_allocated"):
            # The returned data is decoded at the free pointer, which moves past it.
            pointer = self.free_pointer() + (len(outcome.data) + 31) // 32 * 32
            self.write(0x40, (pointer & MASK).to_bytes(32, "big"))
        if call.kind == "eth-transfer":
            if not outcome.success:
                raise _Halt(outcome.data, True)
            return []
        results = [int(outcome.success)]
        targets = stmt.targets if isinstance(stmt, Assign) else ()
        for index in range(1, len(targets)):
            word = outcome.data[32 * (index - 1) : 32 * index].ljust(32, b"\0")
            results.append(int.from_bytes(word, "big"))
        return results

    def check_arguments(
        self, call: ExternalCallExpr, stmt: Stmt, data: bytes, start: int, frame: _Frame
    ) -> None:
        """Printed selector and arguments must match the bytes the call sends."""
        selector = self.selector_of(stmt, required=False)
        if selector is not None and data[:4] != selector:
            raise ViewMismatch(f"call {call.method}: selector differs from its input")
        for index, arg in enumerate(call.args):
            head = data[4 + 32 * index : 36 + 32 * index].ljust(32, b"\0")
            word = int.from_bytes(head, "big")
            if isinstance(arg, Call) and arg.name in DYNAMIC_MARKERS:
                expected = (self.eval(arg.args[0], frame) - start - 4) & MASK
            else:
                expected = self.eval(arg, frame) & MASK
            if word != expected:
                raise ViewMismatch(
                    f"call {call.method}: printed argument {index} differs from its input"
                )

    def selector_of(self, stmt: Stmt, required: bool = True) -> bytes | None:
        if "selector" in stmt.meta:  # decoded from memory by the view itself
            return bytes.fromhex(stmt.meta["selector"][2:])
        info = self.program.context.externals.get(stmt.origin or "")
        if info is None or len(info.selector_candidates) != 1:
            if not required:
                return None
            raise UnsupportedExecution("Call selector unknown")
        return bytes.fromhex(info.selector_candidates[0][2:])

    def emit(self, stmt: Emit, frame: _Frame) -> None:
        origin = self.program.context.statements.get(stmt.origin or "")
        if origin is None:
            raise UnsupportedExecution("Event without origin")
        topic0 = self.program.context.constants.get(origin.operands[2])
        if topic0 is None:
            raise UnsupportedExecution("Event topic is not constant")
        topics = (topic0, *(self.eval(a, frame) for a in stmt.args[: stmt.indexed]))
        rest = stmt.args[stmt.indexed :]
        if "staged-at-free-pointer" in stmt.rules:
            # The data encoded at the free pointer (not moved), as compiled.
            data = self.write_encoding(self.free_pointer(), rest, frame)
        elif len(rest) == 1 and isinstance(rest[0], Slice):
            data = self.slice(rest[0], frame)
        else:
            data = b"".join(self.eval(a, frame).to_bytes(32, "big") for a in rest)
        self.logs.append(RecoveredLog(self.address, tuple(topics), data))

    # --- expressions ---------------------------------------------------------

    def truth(self, expr: Expr, frame: _Frame) -> bool:
        return self.eval(expr, frame) != 0

    def payload(self, payload: Expr | None, frame: _Frame) -> bytes:
        if payload is None:
            return b""
        return self.payload_bytes(payload, frame)

    def payload_bytes(self, payload: Expr, frame: _Frame) -> bytes:
        if isinstance(payload, Env) and payload.name == "returndata":
            return self.returndata
        if isinstance(payload, Slice):
            return self.slice(payload, frame)
        if isinstance(payload, MemoryBytes):
            return self.memory_bytes(payload, frame)
        if (
            isinstance(payload, Call)
            and payload.name == "Error"
            and isinstance(payload.args[0], Raw)
        ):
            import json

            return encode_error(json.loads(payload.args[0].text))
        if isinstance(payload, Call) and payload.name == "Panic":
            code = self.eval(payload.args[0], frame)
            return bytes.fromhex("4e487b71") + code.to_bytes(32, "big")
        if isinstance(payload, CustomError):
            expected = payload.assumed_pointer
            if expected is not None and self.free_pointer() != expected:
                raise ViewMismatch(
                    f"error payload: free pointer {self.free_pointer():#x}, expected {expected:#x}"
                )
            return payload.selector.to_bytes(4, "big") + b"".join(
                self.eval(a, frame).to_bytes(32, "big") for a in payload.args
            )
        if isinstance(payload, Call) and payload.name == "abi.encodeWithSelector":
            selector_text = payload.args[0]
            assert isinstance(selector_text, Raw)
            selector = bytes.fromhex(selector_text.text.split()[0][2:])
            return selector + b"".join(
                self.eval(a, frame).to_bytes(32, "big") for a in payload.args[1:]
            )
        if isinstance(payload, Raw) and payload.text.startswith("0x"):
            return bytes.fromhex(payload.text[2:])
        raise UnsupportedExecution(f"Unsupported payload {payload!r}")

    def slice(self, expr: Slice, frame: _Frame) -> bytes:
        start, end = self.eval(expr.start, frame), self.eval(expr.end, frame)
        if expr.space != "MEM":
            raise UnsupportedExecution("Non-memory slice")
        if end < start:
            raise UnsupportedExecution("Negative memory range")
        return self.read(start, end - start)

    def read(self, start: int, size: int) -> bytes:
        if size == 0:
            return b""
        if start + size > self.max_memory:
            raise UnsupportedExecution("Memory limit exceeded")
        if len(self.memory) < start + size:
            self.memory.extend(b"\0" * (start + size - len(self.memory)))
        return bytes(self.memory[start : start + size])

    def write(self, start: int, data: bytes) -> None:
        if not data:
            return
        self.read(start, len(data))
        self.memory[start : start + len(data)] = data

    def calldata_word(self, offset: int) -> int:
        if offset >= len(self.calldata):
            return 0
        return int.from_bytes(self.calldata[offset : offset + 32].ljust(32, b"\0"), "big")

    def copy(self, call: Call, frame: _Frame) -> None:
        args = [self.eval(a, frame) for a in call.args]
        destination, source, size = args
        if size > self.max_memory:
            raise UnsupportedExecution("Copy limit exceeded")
        if call.name == "CALLDATACOPY":
            data = self.calldata[source : source + size].ljust(size, b"\0")
        elif call.name == "RETURNDATACOPY":
            if source + size > len(self.returndata):
                raise _Halt(b"", True)
            data = self.returndata[source : source + size]
        elif call.name == "MCOPY":
            data = self.read(source, size)
        else:
            code = self.program.context.program.runtime_bytecode
            if code is None:
                raise UnsupportedExecution("CODECOPY requires the exact executed code input")
            runtime = bytes.fromhex(code)
            data = runtime[source : source + size].ljust(size, b"\0")
        self.write(destination, data)

    def storage_slot(self, expr: Expr, frame: _Frame) -> tuple[int, int, int]:
        if isinstance(expr, Var) and expr.ssa is None:
            root = self.program.storage_root(expr.name)
            if root is None:
                raise UnsupportedExecution(f"Unknown storage root {expr.name}")
            location, slot, low, high = root
            if location != "storage":
                raise UnsupportedExecution("Transient storage is not modelled")
            return int(slot, 16), low, high
        if isinstance(expr, Index) and expr.kind == "mapping":
            parent, _, _ = self.storage_slot(expr.base, frame)
            key = self.eval(expr.key, frame)
            return (
                int.from_bytes(
                    keccak256(key.to_bytes(32, "big") + parent.to_bytes(32, "big")), "big"
                ),
                0,
                31,
            )
        if isinstance(expr, Call) and expr.name.startswith("packed_bytes_") and len(expr.args) == 1:
            parent, _, _ = self.storage_slot(expr.args[0], frame)
            low, high = (int(x) for x in expr.name.removeprefix("packed_bytes_").split("_"))
            return parent, low, high
        if isinstance(expr, Index) and expr.kind and ":" in expr.kind:
            kind, stride = expr.kind.split(":")
            parent, _, _ = self.storage_slot(expr.base, frame)
            key = self.eval(expr.key, frame)
            if kind == "array":
                start = int.from_bytes(keccak256(parent.to_bytes(32, "big")), "big")
            else:
                start = parent
            return (start + key * int(stride)) & MASK, 0, 31
        if isinstance(expr, Member) and expr.name == "data":
            parent, _, _ = self.storage_slot(expr.base, frame)
            return int.from_bytes(keccak256(parent.to_bytes(32, "big")), "big"), 0, 31
        if isinstance(expr, Member) and expr.name.startswith("field"):
            parent, _, _ = self.storage_slot(expr.base, frame)
            return (parent + int(expr.name[5:])) & MASK, 0, 31
        raise UnsupportedExecution("Storage path not executable: " + _show(expr))

    def eval(self, expr: Expr, frame: _Frame) -> int:
        if isinstance(expr, Const):
            return expr.value & MASK
        if isinstance(expr, Var):
            if expr.name in frame.values:
                return frame.values[expr.name]
            if expr.ssa is None:
                slot, low, high = self.storage_slot(expr, frame)
                word = self.storage.get(slot, 0)
                return (word >> (8 * low)) & ((1 << (8 * (high - low + 1))) - 1)
            raise UnsupportedExecution(f"Unassigned local {expr.name}")
        if isinstance(expr, Env):
            return self.environment(expr.name)
        if isinstance(expr, Unary):
            value = self.eval(expr.operand, frame)
            if expr.op == "!":
                return int(value == 0)
            if expr.op == "~":
                return MASK ^ value
            if expr.op == "-":
                return (-value) & MASK
        if isinstance(expr, Binary):
            return self.binary(expr, frame)
        if isinstance(expr, Call):
            return self.call_value(expr, frame)
        if isinstance(expr, Index):
            if isinstance(expr.base, Env) and expr.base.name == "msg.data":
                return self.calldata_word(self.eval(expr.key, frame))
            if isinstance(expr.base, Env) and expr.base.name == "MEM":
                return int.from_bytes(self.read(self.eval(expr.key, frame), 32), "big")
            if isinstance(expr.base, Env) and expr.base.name == "STORAGE":
                return self.storage.get(self.eval(expr.key, frame), 0)
            slot, _, _ = self.storage_slot(expr, frame)
            return self.storage.get(slot, 0)
        if isinstance(expr, Field):
            return int.from_bytes(self.read(self.eval(expr.base, frame) + expr.offset, 32), "big")
        if isinstance(expr, Element):
            if expr.space == "msg.data":
                return self.calldata_word(self.element(expr, frame))
            return int.from_bytes(self.read(self.element(expr, frame), 32), "big")
        if isinstance(expr, Member):
            path = _member_path(expr)
            if path is not None and path in frame.values:
                return frame.values[path]
            if (
                expr.name == "length"
                and isinstance(expr.base, Env)
                and expr.base.name == "returndata"
            ):
                return len(self.returndata)
            if expr.name == "length" and isinstance(expr.base, Member) and expr.base.name == "code":
                target = self.eval(expr.base.base, frame) & ADDRESS
                runtime = self.program.context.program.runtime_bytecode
                if target == self.address and runtime:
                    return len(bytes.fromhex(runtime))
                if target not in self.code_sizes:
                    raise UnsupportedExecution("Code size requires an observed value")
                return self.code_sizes[target]
            if (
                expr.name in {"length", "data", "offset"}
                and isinstance(expr.base, Var)
                and expr.base.name in frame.values
            ):
                # Calldata accessors of a dynamic argument (its value is 4 + ABI offset).
                pointer = frame.values[expr.base.name]
                if expr.name == "offset":
                    return (pointer - 4) & MASK
                return (
                    self.calldata_word(pointer) if expr.name == "length" else (pointer + 32) & MASK
                )
            if expr.name.startswith("field") or expr.name == "data":
                slot, _, _ = self.storage_slot(expr, frame)
                return self.storage.get(slot, 0)
            if expr.name == "length" and isinstance(expr.base, (Var, Index, Member)):
                slot, _, _ = self.storage_slot(expr.base, frame)
                return self.storage.get(slot, 0)
            if expr.name == "balance":
                return self.balance(expr.base, frame)
            if expr.name == "length" and isinstance(expr.base, Member) and expr.base.name == "code":
                target = self.eval(expr.base.base, frame) & ADDRESS
                if target not in self.code_sizes:
                    raise UnsupportedExecution("Code size requires an observed value")
                return self.code_sizes[target]
        raise UnsupportedExecution("Unsupported expression: " + _show(expr))

    def balance(self, base: Expr, frame: _Frame) -> int:
        target = self.eval(base, frame) & ADDRESS
        try:
            _op, observed_target, value = next(self.balances)
        except StopIteration as exc:
            raise UnsupportedExecution("Balance requires an observed value") from exc
        if observed_target != target:
            raise UnsupportedExecution("Observed balance target disagrees")
        return value

    def environment(self, name: str) -> int:
        if name == "msg.sender":
            return self.caller
        if name == "msg.value":
            return self.callvalue
        if name == "address(this)":
            return self.address
        if name == "msg.data.length":
            return len(self.calldata)
        if name == "msg.sig":
            return self.calldata_word(0) >> 224
        if name == "true":
            return 1
        if name == "false":
            return 0
        if name == "tx.origin":
            # As in the IR harness: the external caller originates the transaction.
            return self.caller
        if name in self.block_environment:
            return self.block_environment[name]
        opcode = ENVIRONMENT_OPCODES.get(name)
        if opcode is not None and opcode in self.block_environment:
            return self.block_environment[opcode] & MASK
        if name == "block.prevrandao" and "DIFFICULTY" in self.block_environment:
            return self.block_environment["DIFFICULTY"] & MASK
        raise UnsupportedExecution(f"Environment value {name} needs an observation")

    def call_value(self, expr: Call, frame: _Frame) -> int:
        name = expr.name
        if name == "keccak256":
            (data,) = expr.args
            if isinstance(data, AbiCodec) and data.op == "encodePacked":
                return int.from_bytes(keccak256(self.encode_packed(data, frame)), "big")
            if isinstance(data, AbiCodec) and data.op == "encode":
                return int.from_bytes(keccak256(self.encode_tuple(data, frame)), "big")
            if isinstance(data, MemoryBytes):
                if data.length is not None:
                    # The length the allocation wrote must still be the length word.
                    pointer = self.eval(data.pointer, frame)
                    word = int.from_bytes(self.read(pointer, 32), "big")
                    if word != self.eval(data.length, frame):
                        raise UnsupportedExecution("Hashed bytes length word changed")
                return int.from_bytes(keccak256(self.memory_bytes(data, frame)), "big")
            assert isinstance(data, Slice)
            return int.from_bytes(keccak256(self.slice(data, frame)), "big")
        if name.startswith("packed_bytes_") and len(expr.args) == 1:
            slot, low, high = self.storage_slot(expr, frame)
            word = self.storage.get(slot, 0)
            return (word >> (8 * low)) & ((1 << (8 * (high - low + 1))) - 1)
        if name in self.functions:
            results = self.private(expr, frame)
            if len(results) != 1:
                raise UnsupportedExecution("Private call used as a single value")
            return results[0]
        if name == "phi":
            raise UnsupportedExecution("PHI inputs are not associated with predecessors")
        if name == "word" and len(expr.args) == 2:
            return self.calldata_word(
                (self.eval(expr.args[0], frame) + self.eval(expr.args[1], frame)) & MASK
            )
        if name == "blockhash":
            number = self.eval(expr.args[0], frame)
            if number not in self.block_hashes:
                raise UnsupportedExecution("BLOCKHASH requires an observed value")
            return self.block_hashes[number] & MASK
        if name == "gasleft":
            raise UnsupportedExecution("gasleft() needs an observed value")
        args = [self.eval(a, frame) for a in expr.args]
        if name == "byte":
            index, word = args
            return (word >> (8 * (31 - index))) & 0xFF if index < 32 else 0
        if name == "signextend":
            from ..validation import arithmetic

            return arithmetic("SIGNEXTEND", args)
        if name == "addmod":
            return (args[0] + args[1]) % args[2] if args[2] else 0
        if name == "mulmod":
            return (args[0] * args[1]) % args[2] if args[2] else 0
        if len(args) == 1:
            return _cast(name, args[0])
        raise UnsupportedExecution(f"Unsupported call {name}")

    def binary(self, expr: Binary, frame: _Frame) -> int:
        op = expr.op
        if op == "&&":
            return int(self.truth(expr.left, frame) and self.truth(expr.right, frame))
        if op == "||":
            return int(self.truth(expr.left, frame) or self.truth(expr.right, frame))
        a, b = self.eval(expr.left, frame), self.eval(expr.right, frame)
        if op == "+":
            return (a + b) & MASK
        if op == "-":
            return (a - b) & MASK
        if op == "*":
            return (a * b) & MASK
        if op == "/":
            return a // b if b else 0
        if op == "%":
            return a % b if b else 0
        if op == "**":
            return pow(a, b, 1 << 256)
        if op == "&":
            return a & b
        if op == "|":
            return a | b
        if op == "^":
            return a ^ b
        if op == "<<":
            return (a << b) & MASK if b < 256 else 0
        if op == ">>":
            return a >> b if b < 256 else 0
        if op == "s>>":
            return (signed(a) >> min(b, 255)) & MASK
        if op == "==":
            return int(a == b)
        if op == "!=":
            return int(a != b)
        if op == "<":
            return int(a < b)
        if op == ">":
            return int(a > b)
        if op == "<=":
            return int(a <= b)
        if op == ">=":
            return int(a >= b)
        if op == "s<":
            return int(signed(a) < signed(b))
        if op == "s>":
            return int(signed(a) > signed(b))
        if op == "s<=":
            return int(signed(a) <= signed(b))
        if op == "s>=":
            return int(signed(a) >= signed(b))
        if op == "s/":
            if b == 0:
                return 0
            quotient = abs(signed(a)) // abs(signed(b))
            return (quotient if (signed(a) < 0) == (signed(b) < 0) else -quotient) & MASK
        if op == "s%":
            if b == 0:
                return 0
            remainder = abs(signed(a)) % abs(signed(b))
            return (remainder if signed(a) >= 0 else -remainder) & MASK
        raise UnsupportedExecution(f"Unsupported operator {op}")


def describe_trace(statements: list[Stmt], limit: int = 12) -> str:
    """Last executed statements, printed as in the view (for divergence reports)."""
    from .printer import Printed, print_statement

    lines = []
    for stmt in statements[-limit:]:
        out = Printed()
        if isinstance(stmt, (If, While)):
            from .printer import expression

            head = "if" if isinstance(stmt, If) else "while"
            condition = expression(stmt.condition) if stmt.condition is not None else "true"
            lines.append(f"{head} ({condition}) [origin {stmt.origin}]")
            continue
        print_statement(out, stmt, 0)
        lines.append(out.lines[0].strip() + f" [origin {stmt.origin}]")
    return "\n".join(lines)
