# Held-out corpus `heldout-4-2026-10-18`

Eight verified mainnet runtimes, chosen and committed on 2026-10-18 before any
of them was analyzed, after the raw-share work stopped at b0bab72. Sets 1-3
(`fixtures/evaluation-heldout*/`) are calibration now: fixes have been made
while looking at them, so only this set measures how the work carries over.
Bytecode and ABI come from Sourcify (`scripts/fetch-heldout-corpus.py`); the
ABI is scoring ground truth and is never given to the decompiler.

| id | Compiler | Pipeline | Bytes | Why |
| --- | --- | --- | --- | --- |
| entrypoint-v06 | solc 0.8.17 | via-IR, 1e6 runs | 23,689 | ERC-4337: calldata struct arrays, assembly, try/catch |
| universal-router | solc 0.8.17 | via-IR, 1e6 runs | 17,958 | command-byte dispatch, abi.decode of calldata slices |
| morpho-blue | solc 0.8.19 | via-IR, 999999 runs | 15,623 | struct parameters, callbacks with bytes, EIP-712 |
| uni-token | solc 0.5.16 | legacy, 999999 runs | 12,567 | governance token: checkpoints, delegateBySig, uint96 |
| lido-steth-impl | solc 0.4.24 | legacy, 200 runs | 23,361 | AragonApp: unstructured storage, shares accounting |
| yearn-v3-vault | Vyper 0.3.7 | - | 23,754 | ERC-4626, dynamic arrays; a Vyper version not seen before |
| comet-usdc-proxy | solc 0.8.15 | via-IR, runs 1, custom Yul steps | 1,878 | transparent proxy: admin checks, delegatecall |
| mkr-token | solc 0.4.18 | legacy, 200 runs | 3,379 | DSToken: ds-auth authority checks, bytes32 metadata |

None of these runtimes appears in any earlier corpus or fixture (MetaMorpho's
runtime embeds Morpho Blue's address as a call target; the bytecode differs).

Rules: the first native scores are committed before any code change made
because of them; Dedaub exports (requested in `dedaub-exports/REQUESTED.md`,
kept local) are compared only after those scores are recorded.
