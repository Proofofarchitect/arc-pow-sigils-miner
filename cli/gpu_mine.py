#!/usr/bin/env python3
"""
gpu_mine.py — Arc PoW NFT CUDA miner (a standalone CUDA keccak PoW miner).

Formula (PowMintNFTv3.sol, verified by mining/gpu/verify_vector.py):
    work = keccak256(abi.encodePacked(block.chainid, address(this), miner, nonce))
    valid ⟺ uint256(work) < targetFor(miner)   # v3.4; bits-equivalent when no fractional discount
    effective bits = baseBits + 2*epochIndex + loadAdjust + streakBits − stakingDiscount(milli-bits)

Preimage = uint256(chainId) ‖ contract(20) ‖ miner(20) ‖ uint256(nonce) = 104 bytes,
which is exactly ONE Keccak rate block (rate = 136), so the kernel hashes a single
absorption round — same speed profile as a standard single-round keccak kernel.

See README.md for the preimage layout, the two GPU-port traps, and usage.
Two silent GPU-port traps APPLY here and are
handled exactly as in the original: the nonce words are byte-swapped into the little-endian
Keccak lanes, and the leading-zero test runs on the byte-swapped top lane (bswap64).

Arc fee rule baked in: maxFeePerGas >= 20 gwei, otherwise the tx is silently dropped
by the Arc mempool.

Wallet handling is identical to the original: key comes from --pk / $PRIVATE_KEY, is only
ever fed to Account.from_key(), and is NEVER printed. Do not paste keys on the command line
in shared shells; prefer a file or env.

Install on a rented GPU box:
    pip install --break-system-packages cupy-cuda12x eth-account eth-hash[pycryptodome]
Run (dry, no broadcast):
    PRIVATE_KEY=0x... python3 gpu_mine.py --gpus 0 --dry-run
Run (live mainnet):
    PRIVATE_KEY=0x... python3 gpu_mine.py --gpus 0,1,2,3
"""
import argparse
import http.client
import json
import os
import queue
import ssl
import sys
import threading
import time

import numpy as np
import cupy as cp
from eth_hash.auto import keccak
from eth_account import Account

# ------------------------------------------------------------------ config
RPC_DEFAULT = "https://rpc.mainnet.arc.io"
# PowMintNFTv3.4 "Proof of Architect" mainnet instance (testnet stack paused).
# Override per-run with --contract (and --chain-id/--rpc for mainnet).
CONTRACT = "0x3E20bb7be2C46f94Cab78d340D3F79Afc2a9Fed4"
CHAIN_ID = 5042
MIN_FEE_WEI = 20 * 10**9            # Arc mempool floor: below this the tx is silently dropped
DEFAULT_PRIORITY_WEI = 10**9
DEFAULT_GAS_LIMIT = 400000
MAX_HITS = 64
UA = {"content-type": "application/json",
      "user-agent": "Mozilla/5.0 (X11; Linux x86_64) arc-pow-miner/1.0"}

# Optional comma-separated extra RPCs (same chain!): e.g. a private node + public.
RPC_FALLBACKS = [u for u in os.environ.get("ARC_RPC_FALLBACKS", "").split(",") if u]

# ------------------------------------------------------------------ CUDA kernel
# Keccak-f[1600] over the 136-byte padded block. Lanes 0..16 hold the constant
# preimage; only lanes 11 and 12 (bytes 88..103 = high/low 64-bit nonce words) vary.
KERNEL = r"""
typedef unsigned long long u64;
typedef unsigned int u32;

__device__ constexpr u64 RC[24] = {
 0x0000000000000001ULL,0x0000000000008082ULL,0x800000000000808aULL,0x8000000080008000ULL,
 0x000000000000808bULL,0x0000000080000001ULL,0x8000000080008081ULL,0x8000000000008009ULL,
 0x000000000000008aULL,0x0000000000000088ULL,0x0000000080008009ULL,0x000000008000000aULL,
 0x000000008000808bULL,0x800000000000008bULL,0x8000000000008089ULL,0x8000000000008003ULL,
 0x8000000000008002ULL,0x8000000000000080ULL,0x000000000000800aULL,0x800000008000000aULL,
 0x8000000080008081ULL,0x8000000000008080ULL,0x0000000080000001ULL,0x8000000080008008ULL};
__device__ constexpr int RHO[24] = {1,3,6,10,15,21,28,36,45,55,2,14,27,41,56,8,25,43,62,18,39,61,20,44};
__device__ constexpr int PIL[24] = {10,7,11,17,18,3,5,16,8,21,24,4,15,23,19,13,12,2,20,14,22,9,6,1};

#define ROL64(x,n) (((x) << (n)) | ((x) >> (64-(n))))
__device__ __forceinline__ u32 bswap32_(u32 x) { return (x >> 24) | ((x >> 8) & 0xff00u) | ((x << 8) & 0xff0000u) | (x << 24); }
__device__ __forceinline__ u64 bswap64_(u64 x) { return ((u64)bswap32_((u32)x) << 32) | (u64)bswap32_((u32)(x >> 32)); }

__device__ __forceinline__ void keccakf(u64 st[25]) {
  #pragma unroll
  for (int r = 0; r < 24; r++) {
    u64 bc[5];
    #pragma unroll
    for (int i = 0; i < 5; i++) bc[i] = st[i] ^ st[i+5] ^ st[i+10] ^ st[i+15] ^ st[i+20];
    #pragma unroll
    for (int i = 0; i < 5; i++) {
      u64 t = bc[(i+4)%5] ^ ROL64(bc[(i+1)%5], 1);
      #pragma unroll
      for (int j = 0; j < 25; j += 5) st[j+i] ^= t;
    }
    u64 t = st[1];
    #pragma unroll
    for (int i = 0; i < 24; i++) { int j = PIL[i]; u64 tmp = st[j]; st[j] = ROL64(t, RHO[i]); t = tmp; }
    #pragma unroll
    for (int j = 0; j < 25; j += 5) {
      u64 b0 = st[j], b1 = st[j+1], b2 = st[j+2], b3 = st[j+3], b4 = st[j+4];
      st[j]   = b0 ^ ((~b1) & b2);
      st[j+1] = b1 ^ ((~b2) & b3);
      st[j+2] = b2 ^ ((~b3) & b4);
      st[j+3] = b3 ^ ((~b4) & b0);
      st[j+4] = b4 ^ ((~b0) & b1);
    }
    st[0] ^= RC[r];
  }
}

// nonce = (hi64 << 64) | counter ; written big-endian into bytes 72..103,
// i.e. lanes 11 (bytes 88..95) and 12 (bytes 96..103) of the little-endian state,
// so each word must be byte-swapped (bswap64_) before XOR into the lane.
extern "C" __global__ void mine_kernel(const u64* __restrict__ base, u64 counter0, u64 hi64,
                            u64 target_top, u64* __restrict__ hits, u32* __restrict__ hit_count,
                            u32 max_hits, u32 iters) {
  u64 b[17];
  #pragma unroll
  for (int i = 0; i < 17; i++) b[i] = base[i];       // constant preimage in registers
  const u64 hi_swap = bswap64_(hi64);
  u64 stride = (u64)gridDim.x * blockDim.x;
  u64 c = counter0 + (u64)blockIdx.x * blockDim.x + threadIdx.x;
  #pragma unroll 1
  for (u32 it = 0; it < iters; it++, c += stride) {
    u64 st[25];
    #pragma unroll
    for (int i = 0; i < 25; i++) st[i] = 0;
    #pragma unroll
    for (int i = 0; i < 17; i++) st[i] = b[i];
    st[11] = hi_swap;
    st[12] = bswap64_(c);
    keccakf(st);
    u64 top = bswap64_(st[0]);      // big-endian top 64 bits of the digest
    if (top <= target_top) {
      u32 idx = atomicAdd(hit_count, 1u);
      if (idx < max_hits) hits[idx] = c;
    }
  }
}

extern "C" __global__ void selftest_kernel(const u64* base, u64* out) {
  u64 st[25];
  #pragma unroll
  for (int i = 0; i < 25; i++) st[i] = 0;
  #pragma unroll
  for (int i = 0; i < 17; i++) st[i] = base[i];
  keccakf(st);
  #pragma unroll
  for (int i = 0; i < 4; i++) out[i] = st[i];
}
"""

# ------------------------------------------------------------------ rpc
RPC = RPC_DEFAULT
_tl = threading.local()


def _conn_for(host):
    pool = getattr(_tl, "conns", None)
    if pool is None:
        pool = _tl.conns = {}
    c = pool.get(host)
    if c is None:
        c = http.client.HTTPSConnection(host, timeout=20, context=ssl.create_default_context())
        pool[host] = c
    return c


def _post(url, body):
    rest = url.split("://", 1)[-1]
    host, _, path = rest.partition("/")
    path = "/" + path
    last = None
    for _ in (0, 1):
        c = _conn_for(host)
        try:
            c.request("POST", path, body=body, headers=UA)
            r = c.getresponse()
            data = r.read()
            if r.status != 200:
                raise RuntimeError(f"http {r.status}")
            return json.loads(data)
        except Exception as e:
            last = e
            try:
                c.close()
            except Exception:
                pass
            _tl.conns.pop(host, None)
    raise RuntimeError(f"post {host}: {type(last).__name__} {last}")


def _all_urls():
    return [RPC] + [u for u in RPC_FALLBACKS if u != RPC]


def eth_call_item(data):
    return ("eth_call", [{"to": CONTRACT, "data": data}, "latest"])


def batch_rpc(calls):
    body = json.dumps([{"jsonrpc": "2.0", "id": i, "method": m, "params": p}
                       for i, (m, p) in enumerate(calls)]).encode()
    last = None
    for _ in range(3):
        for url in _all_urls():
            try:
                res = _post(url, body)
                out = [None] * len(calls)
                for item in res:
                    out[item["id"]] = item["result"]
                return out
            except Exception as e:
                last = f"{url}: {type(e).__name__} {e}"
                time.sleep(0.1)
    raise RuntimeError(f"batch failed ({last})")


def rpc(method, params):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    last = None
    for _ in range(3):
        for url in _all_urls():
            try:
                return _post(url, body)["result"]
            except Exception as e:
                last = f"{url}: {type(e).__name__} {e}"
                time.sleep(0.15)
    raise RuntimeError(f"rpc {method} failed ({last})")


def sel(sig):
    return "0x" + keccak(sig.encode()).hex()[:8]


def ts():
    return time.strftime("%H:%M:%S")


# ------------------------------------------------------------------ preimage / pow logic
def build_block(chain_id, contract, miner, nonce):
    """136-byte Keccak block: packed preimage (104 B) + Keccak padding (0x01 … 0x80)."""
    b = bytearray(136)
    b[0:32] = int(chain_id).to_bytes(32, "big")
    b[32:52] = bytes.fromhex(contract[2:].rjust(40, "0"))
    b[52:72] = bytes.fromhex(miner[2:].rjust(40, "0"))
    b[72:104] = int(nonce).to_bytes(32, "big")
    b[104] = 0x01
    b[135] |= 0x80
    return b


def preimage(chain_id, contract, miner, nonce):
    return (int(chain_id).to_bytes(32, "big")
            + bytes.fromhex(contract[2:].rjust(40, "0"))
            + bytes.fromhex(miner[2:].rjust(40, "0"))
            + int(nonce).to_bytes(32, "big"))


def leading_zero_bits(x):
    """Port of PowMintNFT._leadingZeroBits (zero -> 256)."""
    if x == 0:
        return 256
    return 256 - x.bit_length()


def target_top_of(bits):
    """Top-64-bit threshold for >= `bits` leading zeros (exact for bits <= 64)."""
    if bits <= 0:
        return (1 << 64) - 1
    if bits > 64:
        raise ValueError(f"bits={bits} > 64: kernel uses a 64-bit top-word test")
    return (1 << (64 - bits)) - 1


def nonce_of(hi64, counter):
    return (hi64 << 64) | counter


def get_job(miner):
    bits_hex, due_hex, minted_hex, nonce_hex, gas_hex, bal_hex = batch_rpc([
        eth_call_item(sel("requiredBits(address)") + miner[2:].rjust(64, "0")),
        # v3.2 core: mint is payable at price+fee; currentMintDue() returns (due, fee).
        eth_call_item(sel("currentMintDue()")),
        eth_call_item(sel("totalMinted()")),
        ("eth_getTransactionCount", [miner, "pending"]),
        ("eth_gasPrice", []),
        ("eth_getBalance", [miner, "latest"]),
    ])
    return dict(bits=int(bits_hex, 16), price=int(due_hex[:66], 16), minted=int(minted_hex, 16),
                tx_nonce=int(nonce_hex, 16), gas=int(gas_hex, 16), balance=int(bal_hex, 16))


# ------------------------------------------------------------------ submit
def normalize_raw(signed):
    raw = getattr(signed, "raw_transaction", None)
    if raw is None:
        raw = getattr(signed, "rawTransaction", None)
    hx = raw.hex() if isinstance(raw, (bytes, bytearray)) else str(raw)
    if hx.startswith("0x"):
        hx = hx[2:]
    return "0x" + hx


def send_raw(raw, urls):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "eth_sendRawTransaction",
                       "params": [raw]}).encode()
    out, ev = [], threading.Event()

    def go(u):
        try:
            res = _post(u, body)
            h = res.get("result")
            if h and not ev.is_set():
                out.append(h)
                ev.set()
        except Exception:
            pass

    for u in urls[1:]:
        threading.Thread(target=go, args=(u,), daemon=True).start()
    try:
        res = _post(urls[0], body)
        h = res.get("result")
        if h:
            return h
    except Exception:
        pass
    ev.wait(6)
    if out:
        return out[0]
    raise RuntimeError("sendRaw: no RPC accepted")


def submit(acct, nonce, price, net, tries=3):
    """One network round-trip in the happy path. Arc fee rule enforced here."""
    data = sel("mint(uint256)") + int(nonce).to_bytes(32, "big").hex()
    urls = _all_urls()
    last = None
    for _ in range(tries):
        tx_nonce = net.get("nonce")
        if tx_nonce is None:
            tx_nonce = int(rpc("eth_getTransactionCount", [acct.address, "pending"]), 16)
            net["nonce"] = tx_nonce
        gas_price = net.get("gas") or MIN_FEE_WEI
        max_fee = max(gas_price * 2, MIN_FEE_WEI)          # >= 20 gwei (Arc mempool floor)
        max_prio = min(max_fee, max(DEFAULT_PRIORITY_WEI, gas_price // 10))
        tx = {
            "type": 2, "chainId": CHAIN_ID, "to": CONTRACT, "value": int(price), "data": data,
            "nonce": tx_nonce, "gas": DEFAULT_GAS_LIMIT,
            "maxFeePerGas": max_fee, "maxPriorityFeePerGas": max_prio,
        }
        signed = acct.sign_transaction(tx)
        try:
            h = send_raw(normalize_raw(signed), urls)
            net["nonce"] = tx_nonce + 1
            return h
        except Exception as e:
            last = e
            net["nonce"] = None
            time.sleep(0.05)
    raise RuntimeError(f"submit failed: {last}")


# ------------------------------------------------------------------ worker
class GpuWorker(threading.Thread):
    def __init__(self, g, state, hits_q, blocks=4096, threads=256, iters=64):
        super().__init__(daemon=True)
        self.g, self.state, self.q = g, state, hits_q
        self.blocks, self.threads, self.iters = blocks, threads, iters
        # unique 64-bit high word per GPU -> nonces never overlap between cards
        self.hi64 = state["hi_base"] + (g + 1)
        self.counter = int(time.time() * 1e3) & 0x00FFFFFFFFFFFFFF
        self.version = -1

    def run(self):
        with cp.cuda.Device(self.g):
            k = cp.RawKernel(KERNEL, "mine_kernel")
            hits = cp.zeros(MAX_HITS, dtype=cp.uint64)
            cnt = cp.zeros(1, dtype=cp.uint32)
            base_gpu = None
            try:
                while not self.state["stop"]:
                    st = self.state
                    if st["version"] != self.version:
                        self.version = st["version"]
                        base_gpu = cp.asarray(st["base"])
                        self.target_top = st["target_top"]
                    cnt[0] = 0
                    k((self.blocks,), (self.threads,),
                      (base_gpu, cp.uint64(self.counter), cp.uint64(self.hi64),
                       np.uint64(self.target_top), hits, cnt,
                       np.uint32(MAX_HITS), np.uint32(self.iters)))
                    cp.cuda.Stream.null.synchronize()
                    self.counter += self.blocks * self.threads * self.iters
                    st["hashes"][self.g] += self.blocks * self.threads * self.iters
                    n = int(cnt[0].get())
                    if n:
                        for c in [int(x) for x in cp.asnumpy(hits[:min(n, MAX_HITS)])]:
                            self.q.put((self.g, c, self.hi64, self.version, time.time()))
            except Exception as e:
                print(f"[gpu{self.g}] FATAL {type(e).__name__}: {e}", flush=True)


# ------------------------------------------------------------------ main
def main():
    global RPC, CONTRACT, CHAIN_ID
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", help="miner address (defaults to the address of --pk)")
    ap.add_argument("--pk", default=os.environ.get("PRIVATE_KEY"))
    ap.add_argument("--rpc", default=RPC_DEFAULT)
    ap.add_argument("--contract", default=CONTRACT)
    ap.add_argument("--chain-id", type=int, default=CHAIN_ID)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--blocks", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--iters", type=int, default=64, help="kernel iterations per launch")
    ap.add_argument("--dry-run", action="store_true", help="mine + verify, never broadcast")
    ap.add_argument("--job-poll", type=float, default=1.0, help="job refresh interval (s)")
    ap.add_argument("--mint-interval", type=float, default=0.2, help="min seconds between mints")
    ap.add_argument("--max-mints", type=int, default=0, help="stop after N successful broadcasts (0=∞)")
    ap.add_argument("--test-bits", type=int, default=0, help="override difficulty for GPU smoke test")
    a = ap.parse_args()

    RPC = a.rpc
    CONTRACT = a.contract
    CHAIN_ID = a.chain_id
    gpus = [int(x) for x in a.gpus.split(",") if x != ""]

    acct = None
    if a.pk:
        acct = Account.from_key(a.pk)
    address = a.address or (acct.address if acct else None)
    if not address:
        print("[FATAL] need --address or --pk/$PRIVATE_KEY", flush=True)
        sys.exit(2)
    if acct and a.address and acct.address.lower() != a.address.lower():
        print(f"[warn] --address {a.address} != key address {acct.address}", flush=True)
    print(f"[init] {address} gpus={gpus} dry_run={a.dry_run} chain={CHAIN_ID} "
          f"contract={CONTRACT}", flush=True)

    # --- CPU/kernel parity self-test before doing anything ---
    b = build_block(CHAIN_ID, CONTRACT, address, 0)
    base = np.frombuffer(bytes(b[:136]), dtype=np.uint64).copy()
    out = cp.zeros(4, dtype=cp.uint64)
    cp.RawKernel(KERNEL, "selftest_kernel")((1,), (1,), (cp.asarray(base), out))
    got = out.get().tobytes()
    want = keccak(bytes(b[:104]))          # keccak256 of the 104-byte preimage
    if got != want:
        print("[FATAL] self-test FAILED (kernel != python keccak256)\n got ", got.hex(),
              "\n want", want.hex(), flush=True)
        sys.exit(1)
    print("[init] self-test OK (kernel == python keccak256 of preimage)", flush=True)

    job = get_job(address)
    print(f"[job] bits={job['bits']} price={job['price']/1e18:.6f} USDC minted={job['minted']} "
          f"balance={job['balance']/1e18:.6f} gas={job['gas']/1e9:.2f} gwei", flush=True)

    if a.test_bits:
        print(f"[test] difficulty forced to {a.test_bits} bits (smoke test)", flush=True)
    fixed_bits = a.test_bits or None
    init_bits = fixed_bits or job["bits"]
    hi_base = (int(time.time()) << 16) & 0xFFFFFFFFFFFF0000
    state = {
        "base": base, "target_top": target_top_of(init_bits), "version": 0, "stop": False,
        "hashes": {g: 0 for g in gpus}, "hi_base": hi_base, "bits": init_bits,
    }
    hits_q = queue.Queue()
    submit_q = queue.Queue(maxsize=1)
    net = {"acct": acct, "nonce": job["tx_nonce"], "gas": job["gas"], "price": job["price"],
           "balance": job["balance"]}
    stats = {"mints": 0, "sent": 0, "last_submit": 0.0, "last_price": job["price"]}

    # --- background job poller ---
    def poller():
        while not state["stop"]:
            try:
                j = get_job(address)
                cur = net.get("nonce")
                net["nonce"] = j["tx_nonce"] if cur is None else max(cur, j["tx_nonce"])
                net["gas"] = j["gas"]
                net["balance"] = j["balance"]
                stats["last_price"] = j["price"]
                new_bits = fixed_bits or j["bits"]
                if new_bits != state["bits"]:
                    state["bits"] = new_bits
                    state["target_top"] = target_top_of(new_bits)
                    state["version"] += 1
                    print(f"[job] difficulty -> {new_bits} bits (minted={j['minted']})", flush=True)
            except Exception as e:
                print("[warn] job:", e, flush=True)
            time.sleep(a.job_poll)

    # --- serialized submitter ---
    def submitter():
        while not state["stop"]:
            try:
                nonce = submit_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if nonce is None:
                return
            if acct is None:
                print("[hit] valid hit but no key (dry-run / missing --pk)", flush=True)
                continue
            price = stats["last_price"]
            if price > 0 and net["balance"] < price:
                print(f"[warn] balance {net['balance']/1e18:.6f} < price {price/1e18:.6f}; "
                      f"skipping (top up native USDC)", flush=True)
                continue
            try:
                h = submit(acct, nonce, price, net)
                stats["sent"] += 1
                print(f"[sent] {ts()} nonce={nonce} tx={h}", flush=True)
                if a.max_mints and stats["sent"] >= a.max_mints:
                    print(f"[done] reached --max-mints={a.max_mints}", flush=True)
                    state["stop"] = True
            except Exception as e:
                print(f"[send-fail] nonce={nonce}: {e}", flush=True)
            stats["last_submit"] = time.time()

    workers = [GpuWorker(g, state, hits_q, a.blocks, a.threads, a.iters) for g in gpus]
    threading.Thread(target=poller, daemon=True).start()
    threading.Thread(target=submitter, daemon=True).start()
    for w in workers:
        w.start()

    t_last, h_last, t_last_warm = time.time(), 0, time.time()
    try:
        while not state["stop"]:
            # 1) hits: verify each in Python before it can be submitted
            while not hits_q.empty():
                g, c, hi64, ver, t_found = hits_q.get()
                nonce = nonce_of(hi64, c)
                work = keccak(preimage(CHAIN_ID, CONTRACT, address, nonce))
                z = leading_zero_bits(int.from_bytes(work, "big"))
                need = state["bits"]
                ok = z >= need and ver == state["version"]
                if z >= need:
                    print(f"[hit] {ts()} gpu{g} nonce={nonce} bits={z} need={need} "
                          f"ok={ok} +{(time.time()-t_found)*1000:.0f}ms", flush=True)
                if ok and not a.dry_run:
                    if time.time() - stats["last_submit"] >= a.mint_interval:
                        try:
                            submit_q.put_nowait(nonce)
                        except queue.Full:
                            pass
            # 2) keep the primary connection warm
            if time.time() - t_last_warm > 15:
                t_last_warm = time.time()
                try:
                    rpc("eth_blockNumber", [])
                except Exception:
                    pass
            # 3) rate
            now = time.time()
            if now - t_last >= 15:
                hs = sum(state["hashes"].values()) - h_last
                print(f"[rate] {ts()} {hs/(now-t_last)/1e9:.2f} GH/s ({len(gpus)} GPU) "
                      f"sent={stats['sent']}", flush=True)
                t_last, h_last = now, sum(state["hashes"].values())
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n[stop] interrupted", flush=True)
    finally:
        state["stop"] = True


if __name__ == "__main__":
    main()
