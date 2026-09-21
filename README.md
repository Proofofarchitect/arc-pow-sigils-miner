# Arc PoW NFT — miner

Standalone **proof-of-work miner** for the Arc Chain NFT collection
(*Proof of Architect*, ERC-721). You mine a card by finding a `nonce` whose keccak hash
beats the collection's on-chain difficulty target, then submitting `mint(nonce)`.

Two ways to mine:

| Path | Where | Speed | Install |
|---|---|---|---|
| **Browser** (WebGPU / CPU) | the project site `/mine`, or the assets in `browser/` | ~10² MH/s | none |
| **CLI / CUDA** (this repo, `cli/`) | a GPU box (4090/3090…), or a farm | ~GH/s | `pip install -r requirements.txt` |

The browser miner is the zero-friction entry point. The CLI/CUDA miner is the fast path
for anyone setting up hardware — same algorithm, same target, just far more hashes per second.

---

## 1. Formula

```solidity
work = keccak256(abi.encodePacked(chainId, contract, miner, nonce));   // 104-byte preimage
valid ⟺ uint256(work) < targetFor(miner);                              // difficulty target
```

- Preimage is exactly **one Keccak rate block** (104 bytes), so a single absorption round.
- `targetFor(miner)` folds base difficulty, the pace regulator, the per-wallet streak, and the
  staking discount (the discount can be **fractional** — e.g. 0.5 bit — expressed in milli-bits).
- The card's art seed is derived **after** the mint:
  `seed = keccak256(seedOf[id] ‖ blockhash(mintBlockOf[id] + 2))`. Because the block hash does not
  exist when you submit, a miner **cannot pre-select** rare cards — more hashrate just means more
  cards, never better ones. This is why browser and GPU miners are equally fair per card.

> The CLI miner grinds by **leading-zero bits** (`requiredBits`), which is exact for integer
> difficulty and a hair conservative (≤1 bit) when a fractional staking discount is active — every
> solution it finds is valid.

---

## 2. Addresses

| Network | chainId | contract | RPC |
|---|---|---|---|
| Arc **mainnet** (live) | `5042` | `0x3E20bb7be2C46f94Cab78d340D3F79Afc2a9Fed4` | `https://rpc.mainnet.arc.io` |

Override at runtime with `--contract` / `--chain-id` / `--rpc` (the testnet deployment is paused).

---

## 3. CLI / CUDA miner

```bash
pip install -r requirements.txt          # cupy-cuda12x eth-account eth-hash[pycryptodome] numpy

# 0) prove the algorithm on CPU first (no GPU, no key):
cd cli
python3 verify_vector.py                 # offline keccak self-test
python3 cpu_mine.py --bits 12            # low-difficulty reference grind

# 1) dry-run on the GPU — mines + verifies, NEVER broadcasts:
PRIVATE_KEY=0x<your_key> python3 gpu_mine.py --gpus 0 --dry-run

# 2) live:
PRIVATE_KEY=0x<your_key> python3 gpu_mine.py --gpus 0,1,2,3 --max-mints 1
```

On startup the miner runs a **kernel↔Python keccak parity self-test** and exits non-zero if it
fails. Every hit is re-verified in Python before submission.

### Key hygiene (mandatory)

- **Never commit a private key.** Pass it via `$PRIVATE_KEY` or `--pk "$(cat ~/arc.key)"`.
  The miner feeds it only to `Account.from_key()` and **never prints it**.
- Use a **dedicated throw-away wallet** funded with a small amount of native USDC.
- Prefer a key file outside the repo; `keys/`, `*.key` are git-ignored here.

### Arc fee rule (baked in)

`maxFeePerGas ≥ 20 gwei`, otherwise the Arc mempool **silently drops** the tx. Native gas token is
USDC with **18 decimals** (`msg.value` = `currentMintDue()`, which is the wave price plus the mint fee).

---

## 4. Browser miner (`browser/`)

| file | role |
|---|---|
| `miner-gpu-worker.js` | WebGPU compute worker (the fast browser path; batches dispatches, re-verifies on JS) |
| `miner-worker.js` | pure-JS CPU worker (fallback) |
| `miner.js` | wrapper / engine selection |
| `gpu-selftest.html` | standalone WebGPU parity self-test page |

Drop these into any page (or use the project's `/mine` page). The browser path needs no keys in the
repo — the wallet signs in your browser.

---

## 5. Expected rates

| Hardware | Miner | Rate |
|---|---|---|
| RTX 4090 | CLI / CUDA | ~4.5 GH/s |
| RTX 3090 | CLI / CUDA | ~2 GH/s |
| desktop dGPU | browser WebGPU | ~10² MH/s |
| integrated GPU | browser WebGPU | ~12 MH/s |
| CPU (noble keccak) | browser CPU | ~1 MH/s per thread |

The CUDA path is orders of magnitude faster than the browser path — that is expected and cannot be
equalised (browsers have no CUDA, only the slower WebGPU).

---

## 6. Safety

- No keys, credentials, or personal data are stored in this repository. Pass keys at runtime only.
- `verify_vector.py` and `cpu_mine.py` are CPU-only, keyless, and read-only.
- Mine only wallets you control, on networks you are authorised to use.

## License

MIT — see [LICENSE](LICENSE).
