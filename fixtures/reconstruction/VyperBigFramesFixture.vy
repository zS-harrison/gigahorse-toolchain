# Internal calls from functions whose frames are larger than 36 words: Vyper 0.2
# saves the caller's frame (its local variables) around each internal call in a
# loop, which the native rewrite (src/vyper_stack_loops.py) turns into explicit
# loads and stores. Generated: 45 and 55 scalar locals.

stored: public(uint256)


@internal
@view
def _mix(x: uint256, y: uint256) -> uint256:
    return x * 3 + y + self.stored


@internal
@view
def _wide(seed: uint256) -> uint256:
    v0: uint256 = seed * 2
    v1: uint256 = v0 + 1
    v2: uint256 = v1 + 2
    v3: uint256 = v2 + 3
    v4: uint256 = v3 + 4
    v5: uint256 = v4 + 5
    v6: uint256 = v5 + 6
    v7: uint256 = v6 + 7
    v8: uint256 = v7 + 8
    v9: uint256 = v8 + 9
    v10: uint256 = v9 + 10
    v11: uint256 = v10 + 11
    v12: uint256 = v11 + 12
    v13: uint256 = v12 + 13
    v14: uint256 = v13 + 14
    v15: uint256 = v14 + 15
    v16: uint256 = v15 + 16
    v17: uint256 = v16 + 17
    v18: uint256 = v17 + 18
    v19: uint256 = v18 + 19
    v20: uint256 = v19 + 20
    v21: uint256 = v20 + 21
    v22: uint256 = v21 + 22
    v23: uint256 = v22 + 23
    v24: uint256 = v23 + 24
    v25: uint256 = v24 + 25
    v26: uint256 = v25 + 26
    v27: uint256 = v26 + 27
    v28: uint256 = v27 + 28
    v29: uint256 = v28 + 29
    v30: uint256 = v29 + 30
    v31: uint256 = v30 + 31
    v32: uint256 = v31 + 32
    v33: uint256 = v32 + 33
    v34: uint256 = v33 + 34
    v35: uint256 = v34 + 35
    v36: uint256 = v35 + 36
    v37: uint256 = v36 + 37
    v38: uint256 = v37 + 38
    v39: uint256 = v38 + 39
    v40: uint256 = v39 + 40
    v41: uint256 = v40 + 41
    v42: uint256 = v41 + 42
    v43: uint256 = v42 + 43
    v44: uint256 = v43 + 44
    v45: uint256 = v44 + 45
    v46: uint256 = v45 + 46
    v47: uint256 = v46 + 47
    v48: uint256 = v47 + 48
    v49: uint256 = v48 + 49
    v50: uint256 = v49 + 50
    v51: uint256 = v50 + 51
    v52: uint256 = v51 + 52
    v53: uint256 = v52 + 53
    v54: uint256 = v53 + 54
    r: uint256 = self._mix(v0, v54)
    return r + v0 + v7 + v14 + v21 + v28 + v35 + v42 + v49


@external
def big(seed: uint256) -> uint256:
    v0: uint256 = seed + 1
    v1: uint256 = v0 + 1
    v2: uint256 = v1 + 2
    v3: uint256 = v2 + 3
    v4: uint256 = v3 + 4
    v5: uint256 = v4 + 5
    v6: uint256 = v5 + 6
    v7: uint256 = v6 + 7
    v8: uint256 = v7 + 8
    v9: uint256 = v8 + 9
    v10: uint256 = v9 + 10
    v11: uint256 = v10 + 11
    v12: uint256 = v11 + 12
    v13: uint256 = v12 + 13
    v14: uint256 = v13 + 14
    v15: uint256 = v14 + 15
    v16: uint256 = v15 + 16
    v17: uint256 = v16 + 17
    v18: uint256 = v17 + 18
    v19: uint256 = v18 + 19
    v20: uint256 = v19 + 20
    v21: uint256 = v20 + 21
    v22: uint256 = v21 + 22
    v23: uint256 = v22 + 23
    v24: uint256 = v23 + 24
    v25: uint256 = v24 + 25
    v26: uint256 = v25 + 26
    v27: uint256 = v26 + 27
    v28: uint256 = v27 + 28
    v29: uint256 = v28 + 29
    v30: uint256 = v29 + 30
    v31: uint256 = v30 + 31
    v32: uint256 = v31 + 32
    v33: uint256 = v32 + 33
    v34: uint256 = v33 + 34
    v35: uint256 = v34 + 35
    v36: uint256 = v35 + 36
    v37: uint256 = v36 + 37
    v38: uint256 = v37 + 38
    v39: uint256 = v38 + 39
    v40: uint256 = v39 + 40
    v41: uint256 = v40 + 41
    v42: uint256 = v41 + 42
    v43: uint256 = v42 + 43
    v44: uint256 = v43 + 44
    r: uint256 = self._mix(v3, v44)
    return r + v0 + v5 + v10 + v15 + v20 + v25 + v30 + v35 + v40


@external
def bigger(seed: uint256) -> uint256:
    q: uint256 = self._wide(seed)
    self.stored = q % 1000
    return q
