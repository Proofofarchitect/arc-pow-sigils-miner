/* Dependency-free Keccak-256 PoW worker.
   Message protocol:
     { type:"start", chainId, contract, miner, startNonce?, keep?, batchSize? }
     { type:"stop" }
     { type:"selftest" }

   chainId/startNonce accept Number, BigInt, decimal string, or 0x-prefixed string.
*/

"use strict";

const RATE = 136;
const MASK64 = (1n << 64n) - 1n;

// Keccak-f[1600] round constants, low/high 32-bit words.
// NOTE (fix applied by verification harness, 2026-…): the two entries marked
// [FIXED] below were corrupted in the reference artifact. Round 15 must be
// 0x8000000000008003 and round 16 must be 0x8000000000008002. The originals
// (0x80000003 / 0x80008002 low words) broke keccak256("") and all PoW output.
const RC = [
  [0x00000001, 0x00000000], [0x00008082, 0x00000000],
  [0x0000808a, 0x80000000], [0x80008000, 0x80000000],
  [0x0000808b, 0x00000000], [0x80000001, 0x00000000],
  [0x80008081, 0x80000000], [0x00008009, 0x80000000],
  [0x0000008a, 0x00000000], [0x00000088, 0x00000000],
  [0x80008009, 0x00000000], [0x8000000a, 0x00000000],
  [0x8000808b, 0x00000000], [0x0000008b, 0x80000000],
  [0x00008089, 0x80000000],   [0x00008003, 0x80000000], // [FIXED] was [0x80000003, 0x80000000]
  [0x00008002, 0x80000000], // [FIXED] was [0x80008002, 0x80000000]
  [0x00000080, 0x80000000],
  [0x0000800a, 0x00000000], [0x8000000a, 0x80000000],
  [0x80008081, 0x80000000], [0x00008080, 0x80000000],
  [0x80000001, 0x00000000], [0x80008008, 0x80000000]
];

// Rotation offsets, indexed as x + 5*y.
// [FIXED] The reference artifact shipped the TRANSPOSE of the correct rho table
// (e.g. slot 1 held 36 instead of 1). Corrected table below:
const ROT = [
   0,  1, 62, 28, 27,
  36, 44,  6, 55, 20,
   3, 10, 43, 25, 39,
  41, 45, 15, 21,  8,
  18,  2, 61, 56, 14
];

const S = new Array(50).fill(0);
const B = new Array(50).fill(0);
const C = new Array(10).fill(0);
const D = new Array(10).fill(0);

function rolLo(lo, hi, n) {
  if (!n) return lo >>> 0;
  if (n < 32) return ((lo << n) | (hi >>> (32 - n))) >>> 0;
  n -= 32;
  return ((hi << n) | (lo >>> (32 - n))) >>> 0;
}

function rolHi(lo, hi, n) {
  if (!n) return hi >>> 0;
  if (n < 32) return ((hi << n) | (lo >>> (32 - n))) >>> 0;
  n -= 32;
  return ((lo << n) | (hi >>> (32 - n))) >>> 0;
}

// Keccak-f[1600], represented as 25 little-endian 64-bit lanes of u32 pairs.
function keccakF(a) {
  for (let round = 0; round < 24; round++) {
    // Theta
    for (let x = 0; x < 5; x++) {
      let lo = 0, hi = 0;
      for (let y = 0; y < 5; y++) {
        const i = 2 * (x + 5 * y);
        lo ^= a[i];
        hi ^= a[i + 1];
      }
      C[2 * x] = lo >>> 0;
      C[2 * x + 1] = hi >>> 0;
    }

    for (let x = 0; x < 5; x++) {
      const p = (x + 4) % 5;
      const q = (x + 1) % 5;
      D[2 * x] = (C[2 * p] ^ rolLo(C[2 * q], C[2 * q + 1], 1)) >>> 0;
      D[2 * x + 1] = (C[2 * p + 1] ^ rolHi(C[2 * q], C[2 * q + 1], 1)) >>> 0;
    }

    for (let y = 0; y < 5; y++) {
      for (let x = 0; x < 5; x++) {
        const i = 2 * (x + 5 * y);
        a[i] = (a[i] ^ D[2 * x]) >>> 0;
        a[i + 1] = (a[i + 1] ^ D[2 * x + 1]) >>> 0;
      }
    }

    // Rho + Pi: B[y][2x+3y] = ROT(A[x][y]).
    for (let y = 0; y < 5; y++) {
      for (let x = 0; x < 5; x++) {
        const src = 2 * (x + 5 * y);
        const dst = 2 * (y + 5 * ((2 * x + 3 * y) % 5));
        const r = ROT[x + 5 * y];
        B[dst] = rolLo(a[src], a[src + 1], r);
        B[dst + 1] = rolHi(a[src], a[src + 1], r);
      }
    }

    // Chi
    for (let y = 0; y < 5; y++) {
      for (let x = 0; x < 5; x++) {
        const i = 2 * (x + 5 * y);
        const i1 = 2 * (((x + 1) % 5) + 5 * y);
        const i2 = 2 * (((x + 2) % 5) + 5 * y);
        a[i] = (B[i] ^ (~B[i1] & B[i2])) >>> 0;
        a[i + 1] = (B[i + 1] ^ (~B[i1 + 1] & B[i2 + 1])) >>> 0;
      }
    }

    // Iota
    a[0] = (a[0] ^ RC[round][0]) >>> 0;
    a[1] = (a[1] ^ RC[round][1]) >>> 0;
  }
}

function hexByte(v) {
  return (v & 255).toString(16).padStart(2, "0");
}

function bytesToHex(bytes) {
  let out = "";
  for (let i = 0; i < bytes.length; i++) out += hexByte(bytes[i]);
  return out;
}

function hexToBytes(hex, size) {
  hex = String(hex).replace(/^0x/i, "");
  if (!/^[0-9a-fA-F]*$/.test(hex) || hex.length !== size * 2) {
    throw new Error(`Expected a ${size}-byte hexadecimal value`);
  }
  const out = new Uint8Array(size);
  for (let i = 0; i < size; i++) out[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  return out;
}

function uint256Bytes(value) {
  let n = typeof value === "bigint" ? value : BigInt(value);
  if (n < 0n || n > MASK64 ** 4n) throw new Error("Value is outside uint256 range");

  const out = new Uint8Array(32);
  for (let i = 31; i >= 0; i--) {
    out[i] = Number(n & 255n);
    n >>= 8n;
  }
  return out;
}

function readLE32(bytes, offset) {
  return (
    bytes[offset] |
    (bytes[offset + 1] << 8) |
    (bytes[offset + 2] << 16) |
    (bytes[offset + 3] << 24)
  ) >>> 0;
}

function wordHexLE(word) {
  return (
    hexByte(word) +
    hexByte(word >>> 8) +
    hexByte(word >>> 16) +
    hexByte(word >>> 24)
  );
}

function stateDigestHex(state) {
  let out = "0x";
  for (let i = 0; i < 8; i++) out += wordHexLE(state[i]);
  return out;
}

function leadingZeroBits(state) {
  let bits = 0;

  // The digest is squeezed little-endian from lanes, but zero-bit checking uses
  // the resulting digest byte order: byte 0, then byte 1, etc.
  for (let i = 0; i < 8; i++) {
    const word = state[i];
    for (let shift = 0; shift < 32; shift += 8) {
      const byte = (word >>> shift) & 255;
      if (byte === 0) {
        bits += 8;
      } else {
        return bits + Math.clz32(byte) - 24;
      }
    }
  }
  return 256;
}

function incrementNonce(nonce) {
  for (let i = 31; i >= 0; i--) {
    nonce[i] = (nonce[i] + 1) & 255;
    if (nonce[i] !== 0) return true;
  }
  return false; // uint256 overflow
}

// Generic Keccak-256 used by self-test.
function keccak256(bytes) {
  const state = new Array(50).fill(0);
  let offset = 0;

  while (offset + RATE <= bytes.length) {
    for (let i = 0; i < RATE / 4; i++) {
      state[i] ^= readLE32(bytes, offset + i * 4);
    }
    keccakF(state);
    offset += RATE;
  }

  const block = new Uint8Array(RATE);
  block.set(bytes.subarray(offset));
  block[bytes.length - offset] = 0x01; // Keccak padding, not SHA3 padding.
  block[RATE - 1] |= 0x80;

  for (let i = 0; i < RATE / 4; i++) state[i] ^= readLE32(block, i * 4);
  keccakF(state);

  return stateDigestHex(state);
}

let running = false;
let timer = null;
let config = null;

function addCandidate(candidate) {
  const list = config.best;
  list.push(candidate);
  list.sort((a, b) => b.bits - a.bits || a.nonce.localeCompare(b.nonce));
  if (list.length > config.keep) list.length = config.keep;
}

function runBatch() {
  if (!running) return;

  const started = performance.now();
  const found = [];
  const { base, nonce, batchSize } = config;

  for (let count = 0; count < batchSize; count++) {
    for (let i = 0; i < 50; i++) S[i] = base[i];

    // Nonce begins at byte offset 72 = lane 9 and occupies exactly four lanes.
    // [FIXED] artifacts wrote S[18],S[20],...,S[32] (8 strided single words).
    // The 104-byte preimage layout puts nonce bytes 72..103 into state words
    // 18..25 (index k = byte offset k*4). Correct contiguous mapping:
    S[18] = readLE32(nonce, 0);
    S[19] = readLE32(nonce, 4);
    S[20] = readLE32(nonce, 8);
    S[21] = readLE32(nonce, 12);
    S[22] = readLE32(nonce, 16);
    S[23] = readLE32(nonce, 20);
    S[24] = readLE32(nonce, 24);
    S[25] = readLE32(nonce, 28);

    keccakF(S);

    const bits = leadingZeroBits(S);
    if (bits > config.bestBits) config.bestBits = bits;

    const nonceHex = "0x" + bytesToHex(nonce);
    const candidate = {
      nonce: nonceHex,
      hash: stateDigestHex(S),
      bits
    };

    if (config.best.length < config.keep ||
        bits > config.best[config.best.length - 1].bits) {
      addCandidate(candidate);
      found.push(candidate);
    }

    config.attempts++;

    if (!incrementNonce(nonce)) {
      running = false;
      postMessage({
        type: "done",
        reason: "uint256 nonce overflow",
        attempts: config.attempts.toString(),
        best: config.best
      });
      return;
    }
  }

  const now = performance.now();
  const elapsed = now - config.lastProgress;
  if (elapsed >= 250) {
    const totalElapsed = now - config.startedAt;
    const hps = Number(config.attempts) / (totalElapsed / 1000); // [FIXED] was config.attempts / (...): BigInt/Number throws

    postMessage({
      type: "progress",
      attempts: config.attempts.toString(),
      hashesPerSecond: hps,
      bestBits: config.bestBits,
      best: config.best,
      candidates: found
    });

    config.lastProgress = now;
  }

  timer = setTimeout(runBatch, 0);
}

function startMining(message) {
  if (running) stopMining("restarted");

  const chainId = uint256Bytes(message.chainId);
  const contract = hexToBytes(message.contract, 20);
  const miner = hexToBytes(message.miner, 20);
  const nonce = uint256Bytes(message.startNonce ?? 0);

  // Exactly one Keccak rate block: 104 data bytes + Keccak padding.
  const block = new Uint8Array(RATE);
  block.set(chainId, 0);
  block.set(contract, 32);
  block.set(miner, 52);
  block[104] = 0x01;
  block[135] |= 0x80;

  const base = new Array(50).fill(0);
  for (let i = 0; i < RATE / 4; i++) base[i] = readLE32(block, i * 4);

  const now = performance.now();
  config = {
    base,
    nonce,
    keep: Math.max(1, Math.min(Number(message.keep ?? 16), 1000)),
    batchSize: Math.max(128, Math.min(Number(message.batchSize ?? 4096), 262144)),
    attempts: 0n,
    bestBits: -1,
    best: [],
    startedAt: now,
    lastProgress: now
  };

  running = true;
  postMessage({ type: "started", startNonce: "0x" + bytesToHex(nonce) });
  runBatch();
}

function stopMining(reason = "stopped") {
  if (!running) return;
  running = false;
  if (timer) clearTimeout(timer);

  postMessage({
    type: "done",
    reason,
    attempts: config ? config.attempts.toString() : "0",
    best: config ? config.best : []
  });
}

function schemeHash(chainId, contract, miner, nonce) {
  const bytes = new Uint8Array(104);
  bytes.set(uint256Bytes(chainId), 0);
  bytes.set(hexToBytes(contract, 20), 32);
  bytes.set(hexToBytes(miner, 20), 52);
  bytes.set(uint256Bytes(nonce), 72);
  return keccak256(bytes);
}

function selfTest() {
  const tests = [
    {
      name: "keccak256('')",
      actual: keccak256(new Uint8Array()),
      expected: "0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
    },
    {
      name: "scheme nonce=0",
      nonce: 0,
      expected: "0x053254b844d850e77548c330c1c758b8ed6c41213f297aa1e583f149a4215d71"
    },
    {
      name: "scheme nonce=1",
      nonce: 1,
      expected: "0x439569f3d583d13b4c3132bfa7e6ae7fae5f0381696265fe8fff04246c3d5cab"
    },
    {
      name: "scheme nonce=42",
      nonce: 42,
      expected: "0xfe2249316d65f16b298f442ae3ae37d9972a97d5154c37938896db35ca370a69"
    },
    {
      name: "scheme nonce=123456",
      nonce: 123456,
      expected: "0x21a2c3beccf660aadb5ed57fcee9e278e3de4b3f7f38696aa7628eb6ee53d96d"
    }
  ];

  const chainId = 5042;
  const contract = "0x1111111111111111111111111111111111111111";
  const miner = "0x2222222222222222222222222222222222222222";

  let passed = 0;
  for (const test of tests) {
    const actual = test.nonce === undefined
      ? test.actual
      : schemeHash(chainId, contract, miner, test.nonce);

    const ok = actual.toLowerCase() === test.expected.toLowerCase();
    if (ok) passed++;

    postMessage({
      type: "selftest",
      result: ok ? "PASS" : "FAIL",
      name: test.name,
      expected: test.expected,
      actual
    });
  }

  postMessage({
    type: "selftest-summary",
    result: passed === tests.length ? "PASS" : "FAIL",
    passed,
    total: tests.length
  });
}

onmessage = ({ data }) => {
  try {
    if (data.type === "start") startMining(data);
    else if (data.type === "stop") stopMining();
    else if (data.type === "selftest") selfTest();
  } catch (error) {
    running = false;
    postMessage({
      type: "error",
      message: error instanceof Error ? error.message : String(error)
    });
  }
};
