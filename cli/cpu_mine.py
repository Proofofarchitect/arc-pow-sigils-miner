#!/usr/bin/env python3
"""
cpu_mine.py — tiny CPU reference miner for the Arc PoW preimage (NO transactions).

Purpose: prove the grind + difficulty check end-to-end on CPU, at a LOW difficulty
(default 12 bits), using the exact same preimage and the same _leadingZeroBits()
semantics as the on-chain contract (imported from verify_vector.py). It never signs
or sends anything.

This is NOT competitive mining — it exists to sanity-check the algorithm and to give
a deterministic reference rate (~0.17 MH/s in pure Python per project notes).

Usage:
    python3 cpu_mine.py --bits 12
    python3 cpu_mine.py --bits 20 --miner 0x1111...  (takes a few seconds)
"""
import argparse
import sys
import time

from verify_vector import (
    CHAIN_ID, CONTRACT, MINER,
    keccak256, uint256_be, addr20, work_hash,
    leading_zero_bits_solidity, leading_zero_bits_ref,
)


def mine(bits: int, chain_id: int, contract: str, miner: str,
         start_nonce: int = 0, max_nonce: int = None, progress_every: int = 250000):
    """Bit-grind `nonce` until keccak(chainid‖contract‖miner‖nonce) has >= bits leading zeros."""
    prefix = uint256_be(chain_id) + addr20(contract) + addr20(miner)  # 72 bytes
    assert len(prefix) == 72

    t0 = time.time()
    nonce = start_nonce
    attempts = 0
    t_prog = t0
    while max_nonce is None or nonce <= max_nonce:
        h = keccak256(prefix + uint256_be(nonce))
        attempts += 1
        z = leading_zero_bits_solidity(int.from_bytes(h, "big"))
        if z >= bits:
            dt = time.time() - t0
            return nonce, h, z, attempts, dt
        nonce += 1
        if time.time() - t_prog >= 2.0:
            dt = time.time() - t0
            print(f"  ... {attempts} attempts, {attempts/dt/1e6:.2f} MH/s, "
                  f"nonce={nonce}", flush=True)
            t_prog = time.time()
    return None, None, None, attempts, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description="CPU reference Arc PoW miner (no tx).")
    ap.add_argument("--bits", type=int, default=12)
    ap.add_argument("--chain-id", type=int, default=CHAIN_ID)
    ap.add_argument("--contract", default=CONTRACT)
    ap.add_argument("--miner", default=MINER)
    ap.add_argument("--start-nonce", type=int, default=0)
    ap.add_argument("--max-nonce", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0,
                    help="if set, uses a fresh random start nonce (offline demo coin)")
    a = ap.parse_args()

    if a.seed:
        import random
        random.seed(a.seed)
        a.start_nonce = random.getrandbits(64)

    print("=" * 74)
    print(f"CPU reference miner — target {a.bits} leading zero bits (no transactions)")
    print("=" * 74)
    print(f"chainId   : {a.chain_id}")
    print(f"contract  : {a.contract}")
    print(f"miner     : {a.miner}")
    print(f"start     : {a.start_nonce}")

    nonce, h, z, attempts, dt = mine(a.bits, a.chain_id, a.contract, a.miner,
                                     a.start_nonce, a.max_nonce)
    if nonce is None:
        print(f"[fail] no solution within {attempts} attempts")
        return 1

    h_hex = "0x" + h.hex()
    # independent re-check with a second keccak call + the reference lz function
    h2 = work_hash(a.chain_id, a.contract, a.miner, nonce)
    z_ref = leading_zero_bits_ref(int.from_bytes(h2, "big"))
    ok = (h2 == h) and (z_ref >= a.bits) and (z == z_ref)

    print("-" * 74)
    print(f"nonce     : {nonce}")
    print(f"hash      : {h_hex}")
    print(f"lead zeros: {z}  (reference says {z_ref})")
    print(f"attempts  : {attempts}")
    print(f"elapsed   : {dt:.3f}s  ({attempts/dt/1e6:.3f} MH/s)")
    print(f"valid (>= {a.bits}): {'PASS' if ok else 'FAIL'}")
    print("=" * 74)
    print("RESULT:", "PASS" if ok else "FAIL")
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
