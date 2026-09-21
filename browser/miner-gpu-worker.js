/* GPU (WebGPU) Keccak-256 PoW worker for the Proof of Architect miner.
 *
 * Drop-in replacement for `miner-worker.js`: the message protocol is identical,
 * so `lib/miner-client.ts` and the /mine page treat both engines the same:
 *
 *   page -> worker
 *     { type:"start", chainId, contract, miner, startNonce?, keep?, requiredBits?, allowSoftware? }
 *     { type:"stop" }
 *     { type:"selftest", allowSoftware? }
 *
 *   worker -> page
 *     { type:"started",           startNonce, adapter }
 *     { type:"progress",          attempts, hashesPerSecond, bestBits, best, candidates }
 *     { type:"done",              reason, attempts, best }
 *     { type:"error",             message }
 *     { type:"selftest",          result, name, expected, actual }
 *     { type:"selftest-summary",  result, passed, total }                      (JS part)
 *     { type:"gpu-selftest-summary", result, passed, total, adapter, detail? } (WebGPU part)
 *
 * Design notes
 * ------------
 * - The PoW preimage is 104 bytes = exactly ONE Keccak rate block:
 *     keccak256(chainId(32) || contract(20) || miner(20) || nonce(32))
 *   Only state words 18..25 depend on the nonce; everything else (including the
 *   padding bytes 0x01 and 0x80) is constant per job and is uploaded once into a
 *   storage buffer. The kernel is a 1:1 WGSL port of the *proven* JS keccakF
 *   below (u32 words, same tables, same [FIXED] entries).
 * - The GPU never "wins" by itself: every candidate that clears the difficulty
 *   threshold is re-derived with the JS keccak before it is reported to the page
 *   (the page re-verifies again before minting). A mismatch counter disables the
 *   GPU after 3 bad candidates (anti-fault pattern, same idea as hashcats).
 * - `allowSoftware: true` (used only by the self-test page) lifts the software
 *   adapter (SwiftShader/llvmpipe/WARP) rejection so headless CI can validate
 *   the kernel. The production page never sends it.
 */

"use strict";

/* ------------------------------------------------------------------ *
 *  Proven JS Keccak-256 (byte-identical to miner-worker.js) — used for
 *  self-test vectors and for re-verifying every GPU candidate.
 * ------------------------------------------------------------------ */

const RATE = 136;
const MASK64 = (1n << 64n) - 1n;

// Keccak-f[1600] round constants, low/high 32-bit words.
// Two entries below carry the [FIXED] correction from the verification harness:
// round 15 = 0x8000000000008003, round 16 = 0x8000000000008002.
const RC = [
  [0x00000001, 0x00000000], [0x00008082, 0x00000000],
  [0x0000808a, 0x80000000], [0x80008000, 0x80000000],
  [0x0000808b, 0x00000000], [0x80000001, 0x00000000],
  [0x80008081, 0x80000000], [0x00008009, 0x80000000],
  [0x0000008a, 0x00000000], [0x00000088, 0x00000000],
  [0x80008009, 0x00000000], [0x8000000a, 0x00000000],
  [0x8000808b, 0x00000000], [0x0000008b, 0x80000000],
  [0x00008089, 0x80000000], [0x00008003, 0x80000000],
  [0x00008002, 0x80000000], [0x00000080, 0x80000000],
  [0x0000800a, 0x00000000], [0x8000000a, 0x80000000],
  [0x80008081, 0x80000000], [0x00008080, 0x80000000],
  [0x80000001, 0x00000000], [0x80008008, 0x80000000]
];

const ROT = [
   0,  1, 62, 28, 27,
  36, 44,  6, 55, 20,
   3, 10, 43, 25, 39,
  41, 45, 15, 21,  8,
  18,  2, 61, 56, 14
];

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

function keccakF(a) {
  for (let round = 0; round < 24; round++) {
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

    for (let y = 0; y < 5; y++) {
      for (let x = 0; x < 5; x++) {
        const src = 2 * (x + 5 * y);
        const dst = 2 * (y + 5 * ((2 * x + 3 * y) % 5));
        const r = ROT[x + 5 * y];
        B[dst] = rolLo(a[src], a[src + 1], r);
        B[dst + 1] = rolHi(a[src], a[src + 1], r);
      }
    }

    for (let y = 0; y < 5; y++) {
      for (let x = 0; x < 5; x++) {
        const i = 2 * (x + 5 * y);
        const i1 = 2 * (((x + 1) % 5) + 5 * y);
        const i2 = 2 * (((x + 2) % 5) + 5 * y);
        a[i] = (B[i] ^ (~B[i1] & B[i2])) >>> 0;
        a[i + 1] = (B[i + 1] ^ (~B[i1 + 1] & B[i2 + 1])) >>> 0;
      }
    }

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

/** keccak256 over one 136-byte block; returns the raw final state words. */
function keccak256Raw(bytes) {
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
  block[bytes.length - offset] = 0x01;
  block[RATE - 1] |= 0x80;

  for (let i = 0; i < RATE / 4; i++) state[i] ^= readLE32(block, i * 4);
  keccakF(state);

  return state;
}

function keccak256(bytes) {
  return stateDigestHex(keccak256Raw(bytes));
}

/** Build the constant absorb state (nonce words zeroed) for one job. */
function buildBase(chainId, contract, miner) {
  const block = new Uint8Array(RATE);
  block.set(uint256Bytes(chainId), 0);
  block.set(hexToBytes(contract, 20), 32);
  block.set(hexToBytes(miner, 20), 52);
  block[104] = 0x01;
  block[135] |= 0x80;

  const base = new Uint32Array(50);
  for (let i = 0; i < RATE / 4; i++) base[i] = readLE32(block, i * 4);
  return base;
}

/** Re-derive a nonce in JS: { bits, hash } for the given 104-byte scheme. */
function verifyNonce(scheme, nonce) {
  const bytes = new Uint8Array(104);
  bytes.set(scheme.ciBytes, 0);
  bytes.set(scheme.cBytes, 32);
  bytes.set(scheme.mBytes, 52);
  bytes.set(uint256Bytes(nonce), 72);
  const state = keccak256Raw(bytes);
  return { bits: leadingZeroBits(state), hash: stateDigestHex(state) };
}

/* ------------------------------------------------------------------ *
 *  WGSL kernel — direct port of the JS keccakF above.
 * ------------------------------------------------------------------ */

const WGSL = `
struct Params {
  hdr: vec4<u32>,   // start_nonce_lo, start_nonce_hi, total_threads, iters
  cfg: vec4<u32>,   // threshold_bits, mode (0 = grind, 1 = single), reserved, reserved
};

@group(0) @binding(0) var<storage, read> base_state: array<u32, 50>;

@group(0) @binding(1) var<uniform> P: Params;

struct CandBuf {
  count: atomic<u32>,
  entries: array<u32, 256>,   // 64 slots x 4: nonce_lo, nonce_hi, bits, 0
};
@group(0) @binding(2) var<storage, read_write> cands: CandBuf;

const RC: array<u32, 48> = array<u32, 48>(
  0x00000001u, 0x00000000u, 0x00008082u, 0x00000000u,
  0x0000808au, 0x80000000u, 0x80008000u, 0x80000000u,
  0x0000808bu, 0x00000000u, 0x80000001u, 0x00000000u,
  0x80008081u, 0x80000000u, 0x00008009u, 0x80000000u,
  0x0000008au, 0x00000000u, 0x00000088u, 0x00000000u,
  0x80008009u, 0x00000000u, 0x8000000au, 0x00000000u,
  0x8000808bu, 0x00000000u, 0x0000008bu, 0x80000000u,
  0x00008089u, 0x80000000u, 0x00008003u, 0x80000000u,
  0x00008002u, 0x80000000u, 0x00000080u, 0x80000000u,
  0x0000800au, 0x00000000u, 0x8000000au, 0x80000000u,
  0x80008081u, 0x80000000u, 0x00008080u, 0x80000000u,
  0x80000001u, 0x00000000u, 0x80008008u, 0x80000000u
);

const ROT: array<u32, 25> = array<u32, 25>(
  0u, 1u, 62u, 28u, 27u,
  36u, 44u, 6u, 55u, 20u,
  3u, 10u, 43u, 25u, 39u,
  41u, 45u, 15u, 21u, 8u,
  18u, 2u, 61u, 56u, 14u
);

fn bswap32(x: u32) -> u32 {
  return ((x & 0xffu) << 24u) | ((x & 0xff00u) << 8u) |
         ((x & 0xff0000u) >> 8u) | ((x & 0xff000000u) >> 24u);
}

fn rol_lo(lo: u32, hi: u32, n: u32) -> u32 {
  if (n == 0u) { return lo; }
  if (n < 32u) { return (lo << n) | (hi >> (32u - n)); }
  let m = n - 32u;
  if (m == 0u) { return hi; }
  return (hi << m) | (lo >> (32u - m));
}

fn rol_hi(lo: u32, hi: u32, n: u32) -> u32 {
  if (n == 0u) { return hi; }
  if (n < 32u) { return (hi << n) | (lo >> (32u - n)); }
  let m = n - 32u;
  if (m == 0u) { return lo; }
  return (lo << m) | (hi >> (32u - m));
}

fn keccak_f(st: ptr<function, array<u32, 50>>) {
  var c: array<u32, 10>;
  var d: array<u32, 10>;
  var b: array<u32, 50>;

  for (var round: u32 = 0u; round < 24u; round = round + 1u) {
    // Theta
    for (var x: u32 = 0u; x < 5u; x = x + 1u) {
      var lo: u32 = 0u;
      var hi: u32 = 0u;
      for (var y: u32 = 0u; y < 5u; y = y + 1u) {
        let i = 2u * (x + 5u * y);
        lo = lo ^ (*st)[i];
        hi = hi ^ (*st)[i + 1u];
      }
      c[2u * x] = lo;
      c[2u * x + 1u] = hi;
    }

    for (var x: u32 = 0u; x < 5u; x = x + 1u) {
      let p = (x + 4u) % 5u;
      let q = (x + 1u) % 5u;
      d[2u * x] = c[2u * p] ^ rol_lo(c[2u * q], c[2u * q + 1u], 1u);
      d[2u * x + 1u] = c[2u * p + 1u] ^ rol_hi(c[2u * q], c[2u * q + 1u], 1u);
    }

    for (var y: u32 = 0u; y < 5u; y = y + 1u) {
      for (var x: u32 = 0u; x < 5u; x = x + 1u) {
        let i = 2u * (x + 5u * y);
        (*st)[i] = (*st)[i] ^ d[2u * x];
        (*st)[i + 1u] = (*st)[i + 1u] ^ d[2u * x + 1u];
      }
    }

    // Rho + Pi
    for (var y: u32 = 0u; y < 5u; y = y + 1u) {
      for (var x: u32 = 0u; x < 5u; x = x + 1u) {
        let src = 2u * (x + 5u * y);
        let dst = 2u * (y + 5u * ((2u * x + 3u * y) % 5u));
        let r = ROT[x + 5u * y];
        b[dst] = rol_lo((*st)[src], (*st)[src + 1u], r);
        b[dst + 1u] = rol_hi((*st)[src], (*st)[src + 1u], r);
      }
    }

    // Chi
    for (var y: u32 = 0u; y < 5u; y = y + 1u) {
      for (var x: u32 = 0u; x < 5u; x = x + 1u) {
        let i = 2u * (x + 5u * y);
        let i1 = 2u * (((x + 1u) % 5u) + 5u * y);
        let i2 = 2u * (((x + 2u) % 5u) + 5u * y);
        (*st)[i] = b[i] ^ ((~(b[i1])) & b[i2]);
        (*st)[i + 1u] = b[i + 1u] ^ ((~(b[i1 + 1u])) & b[i2 + 1u]);
      }
    }

    // Iota
    (*st)[0u] = (*st)[0u] ^ RC[2u * round];
    (*st)[1u] = (*st)[1u] ^ RC[2u * round + 1u];
  }
}

fn leading_zero_bits(st: ptr<function, array<u32, 50>>) -> u32 {
  var bits: u32 = 0u;
  for (var i: u32 = 0u; i < 8u; i = i + 1u) {
    let word = (*st)[i];
    for (var shift: u32 = 0u; shift < 32u; shift = shift + 8u) {
      let byte = (word >> shift) & 255u;
      if (byte == 0u) {
        bits = bits + 8u;
      } else {
        var v = byte;
        var count: u32 = 0u;
        while ((v & 0x80u) == 0u) {
          count = count + 1u;
          v = v << 1u;
        }
        return bits + count;
      }
    }
  }
  return bits;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let mode = P.cfg.y;
  if (mode == 1u && gid.x != 0u) { return; }

  let total = P.hdr.z;
  let n_iters = select(P.hdr.w, 1u, mode == 1u);
  let threshold = P.cfg.x;

  var base_local: array<u32, 50>;
  for (var i: u32 = 0u; i < 50u; i = i + 1u) { base_local[i] = base_state[i]; }

  var st: array<u32, 50>;

  for (var j: u32 = 0u; j < n_iters; j = j + 1u) {
    let t_local = j * total + gid.x;
    var n_lo = P.hdr.x + t_local;
    let carry = select(0u, 1u, n_lo < P.hdr.x);
    let n_hi = P.hdr.y + carry;

    for (var i: u32 = 0u; i < 50u; i = i + 1u) { st[i] = base_local[i]; }
    // The nonce is 32 big-endian bytes; its last 8 bytes land in state words
    // 24/25 (float: readLE32(nonce, 24) = bswap32(v_hi), readLE32(nonce, 28)
    // = bswap32(v_lo)). Words 18..23 stay zero for a 64-bit counter.
    st[24] = bswap32(n_hi);
    st[25] = bswap32(n_lo);

    keccak_f(&st);

    let bits = leading_zero_bits(&st);
    if (bits >= threshold) {
      let idx = atomicAdd(&cands.count, 1u);
      if (idx < 64u) {
        // entries[] is read back at buffer word (1 + k): slot idx occupies
        // entries[idx*4 .. idx*4+3] — keep the reader's 1 + i*4 offsets aligned.
        let o = idx * 4u;
        cands.entries[o] = n_lo;
        cands.entries[o + 1u] = n_hi;
        cands.entries[o + 2u] = bits;
        cands.entries[o + 3u] = 0u;
      }
    }
  }
}
`;

/* ------------------------------------------------------------------ *
 *  WebGPU plumbing
 * ------------------------------------------------------------------ */

const WORKGROUP = 256;
const GRID = 1024;                       // 262144 threads per dispatch
const TOTAL_THREADS = WORKGROUP * GRID;
const CAND_SLOTS = 64;
const CAND_BYTES = (1 + CAND_SLOTS * 4) * 4; // count + 64 x (lo, hi, bits, pad)
const MIN_ITERS = 16;
const MAX_ITERS = 16384;
const MAX_DISPATCHES = 32;               // dispatches per single sync/readback
const NEAR_MARGIN = 4;                   // report candidates down to required-4
const VERIFY_FAIL_LIMIT = 3;

const SOFTWARE_RE =
  /swiftshader|llvmpipe|lavapipe|software|basic render|warp/i;

let gpu = null;          // { device, pipeline, bindGroup, baseBuf, paramsBuf, candsBuf, stagingBuf, adapterName }
let running = false;
let config = null;       // active mining job
let verifying = false;   // selftest / parity gate in flight

function post(msg) {
  postMessage(msg);
}

function errText(error) {
  return error instanceof Error ? error.message : String(error);
}

async function initGPU(allowSoftware) {
  if (gpu) return;

  const gpuApi = self.navigator ? self.navigator.gpu : undefined;
  if (!gpuApi) throw new Error("WebGPU is not available in this browser");

  let adapter =
    (await gpuApi.requestAdapter({ powerPreference: "high-performance" })) ??
    (await gpuApi.requestAdapter());
  if (!adapter) throw new Error("No WebGPU adapter found");

  const info = adapter.info;
  const desc = [info?.vendor, info?.architecture, info?.device, info?.description]
    .filter(Boolean)
    .join(" ")
    .trim();
  const soft = adapter.isFallbackAdapter === true || SOFTWARE_RE.test(desc);
  if (soft && !allowSoftware) {
    throw new Error("The WebGPU adapter available is a software renderer");
  }

  const device = await adapter.requestDevice();
  device.lost.then((infoLost) => {
    if (running) {
      running = false;
      post({
        type: "error",
        message: `GPU device was lost (${infoLost?.reason ?? "unknown"})`,
      });
    }
  });

  const module = device.createShaderModule({ code: WGSL });
  if (typeof module.getCompilationInfo === "function") {
    try {
      const ci = await module.getCompilationInfo();
      const error = (ci.messages || []).find((m) => m.type === "error");
      if (error) {
        throw new Error(`WGSL compile error: ${error.message} (line ${error.lineNum})`);
      }
    } catch (e) {
      if (e instanceof Error && e.message.startsWith("WGSL")) throw e;
    }
  }

  const pipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });

  const baseBuf = device.createBuffer({
    size: 200,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const paramsBuf = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const candsBuf = device.createBuffer({
    size: CAND_BYTES,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });
  const stagingBuf = device.createBuffer({
    size: CAND_BYTES,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: baseBuf } },
      { binding: 1, resource: { buffer: paramsBuf } },
      { binding: 2, resource: { buffer: candsBuf } },
    ],
  });

  gpu = {
    device,
    pipeline,
    bindGroup,
    baseBuf,
    paramsBuf,
    candsBuf,
    stagingBuf,
    adapterName: desc || "GPU",
  };
}

/** One dispatch of the kernel; reads back candidate entries. */
async function gpuDispatch(nLo, nHi, iters, threshold, mode, grid) {
  const { device, pipeline, bindGroup, paramsBuf, candsBuf, stagingBuf } = gpu;

  const params = new Uint32Array([
    nLo >>> 0,
    nHi >>> 0,
    TOTAL_THREADS,
    iters >>> 0,
    threshold >>> 0,
    mode >>> 0,
    0,
    0,
  ]);
  device.queue.writeBuffer(paramsBuf, 0, params);

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(grid);
  pass.end();
  encoder.copyBufferToBuffer(candsBuf, 0, stagingBuf, 0, CAND_BYTES);
  device.queue.submit([encoder.finish()]);

  await device.queue.onSubmittedWorkDone();
  await stagingBuf.mapAsync(GPUMapMode.READ);
  const mapped = stagingBuf.getMappedRange();
  const words = new Uint32Array(mapped.slice(0));
  stagingBuf.unmap();

  // Reset the counter for the next dispatch (queue-ordered).
  device.queue.writeBuffer(candsBuf, 0, new Uint32Array([0]));

  const count = Math.min(words[0], CAND_SLOTS);
  const entries = [];
  for (let i = 0; i < count; i++) {
    const o = 1 + i * 4;
    entries.push({ nLo: words[o], nHi: words[o + 1], bits: words[o + 2] });
  }
  return { count: words[0], entries };
}

/**
 * `count` dispatches sharing ONE sync + readback. Per-dispatch params and
 * submits are queued (queue-ordered), so each pass receives its own nonce
 * window without a CPU/GPU round-trip between them; the candidate ring
 * accumulates across the batch and is read once. This is the main throughput
 * path — per-dispatch fence+map latency otherwise caps slow-sync browsers at
 * a few MH/s.
 */
async function gpuDispatchBatch(iters, threshold, mode, grid, count) {
  const { device, pipeline, bindGroup, paramsBuf, candsBuf, stagingBuf } = gpu;
  const per = BigInt(TOTAL_THREADS) * BigInt(iters);

  // Reset the ring once (queue-ordered before the first pass).
  device.queue.writeBuffer(candsBuf, 0, new Uint32Array([0]));

  for (let i = 0; i < count; i++) {
    const base = config.base + per * BigInt(i);
    const params = new Uint32Array([
      Number(base & 0xffffffffn) >>> 0,
      Number((base >> 32n) & 0xffffffffn) >>> 0,
      TOTAL_THREADS,
      iters >>> 0,
      threshold >>> 0,
      mode >>> 0,
      0,
      0,
    ]);
    device.queue.writeBuffer(paramsBuf, 0, params);
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(grid);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(candsBuf, 0, stagingBuf, 0, CAND_BYTES);
  device.queue.submit([encoder.finish()]);

  await device.queue.onSubmittedWorkDone();
  await stagingBuf.mapAsync(GPUMapMode.READ);
  const mapped = stagingBuf.getMappedRange();
  const words = new Uint32Array(mapped.slice(0));
  stagingBuf.unmap();

  const cnt = Math.min(words[0], CAND_SLOTS);
  const entries = [];
  for (let i = 0; i < cnt; i++) {
    const o = 1 + i * 4;
    entries.push({ nLo: words[o], nHi: words[o + 1], bits: words[o + 2] });
  }
  return { count: words[0], entries, batch: per * BigInt(count) };
}

/* ------------------------------------------------------------------ *
 *  Self-test: JS vectors (same as miner-worker.js) + WebGPU parity.
 * ------------------------------------------------------------------ */

const SELF_TEST = {
  chainId: 5042,
  contract: "0x" + "11".repeat(20),
  miner: "0x" + "22".repeat(20),
};

// { nonce, bits } — bits precomputed with pycryptodome and cross-checked
// against the JS keccak below at runtime.
const SELF_TEST_VECTORS = [
  { nonce: 123456n, bits: 2 },
  { nonce: 0n, bits: 5 },
  { nonce: 2138n, bits: 15 },
  { nonce: 94904n, bits: 21 },
];

function selfTestScheme() {
  return {
    ciBytes: uint256Bytes(SELF_TEST.chainId),
    cBytes: hexToBytes(SELF_TEST.contract, 20),
    mBytes: hexToBytes(SELF_TEST.miner, 20),
  };
}

async function selfTest(allowSoftware) {
  // --- Part A: the proven JS vectors (kept identical to miner-worker.js). ---
  const jsTests = [
    {
      name: "keccak256('')",
      actual: keccak256(new Uint8Array()),
      expected: "0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470",
    },
    {
      name: "scheme nonce=0",
      nonce: 0,
      expected: "0x053254b844d850e77548c330c1c758b8ed6c41213f297aa1e583f149a4215d71",
    },
    {
      name: "scheme nonce=1",
      nonce: 1,
      expected: "0x439569f3d583d13b4c3132bfa7e6ae7fae5f0381696265fe8fff04246c3d5cab",
    },
    {
      name: "scheme nonce=42",
      nonce: 42,
      expected: "0xfe2249316d65f16b298f442ae3ae37d9972a97d5154c37938896db35ca370a69",
    },
    {
      name: "scheme nonce=123456",
      nonce: 123456,
      expected: "0x21a2c3beccf660aadb5ed57fcee9e278e3de4b3f7f38696aa7628eb6ee53d96d",
    },
  ];

  let passed = 0;
  for (const test of jsTests) {
    let actual;
    if (test.nonce === undefined) {
      actual = test.actual;
    } else {
      const bytes = new Uint8Array(104);
      bytes.set(uint256Bytes(SELF_TEST.chainId), 0);
      bytes.set(hexToBytes(SELF_TEST.contract, 20), 32);
      bytes.set(hexToBytes(SELF_TEST.miner, 20), 52);
      bytes.set(uint256Bytes(test.nonce), 72);
      actual = keccak256(bytes);
    }

    const ok = actual.toLowerCase() === test.expected.toLowerCase();
    if (ok) passed++;
    post({
      type: "selftest",
      result: ok ? "PASS" : "FAIL",
      name: test.name,
      expected: test.expected,
      actual,
    });
  }

  post({
    type: "selftest-summary",
    result: passed === jsTests.length ? "PASS" : "FAIL",
    passed,
    total: jsTests.length,
  });

  // --- Part B: WebGPU parity (kernel vs JS) on layered difficulties. ---
  if (running || verifying) {
    post({
      type: "gpu-selftest-summary",
      result: "SKIP",
      passed: 0,
      total: SELF_TEST_VECTORS.length,
      adapter: null,
      detail: "mining or another test in progress",
    });
    return;
  }

  verifying = true;
  let adapterName = null;
  try {
    await initGPU(!!allowSoftware);
    adapterName = gpu.adapterName;
  } catch (e) {
    verifying = false;
    post({
      type: "gpu-selftest-summary",
      result: "SKIP",
      passed: 0,
      total: SELF_TEST_VECTORS.length,
      adapter: null,
      detail: errText(e),
    });
    return;
  }

  const scheme = selfTestScheme();
  gpu.device.queue.writeBuffer(
    gpu.baseBuf,
    0,
    buildBase(SELF_TEST.chainId, SELF_TEST.contract, SELF_TEST.miner),
  );
  let gpuPassed = 0;
  for (const vector of SELF_TEST_VECTORS) {
    try {
      const nLo = Number(vector.nonce & 0xffffffffn) >>> 0;
      const nHi = Number((vector.nonce >> 32n) & 0xffffffffn) >>> 0;
      const res = await gpuDispatch(nLo, nHi, 1, 0, 1, 1);
      const entry = res.entries[0];
      const jsBits = verifyNonce(scheme, vector.nonce).bits;

      let ok = false;
      let actual = "no candidate returned";
      if (entry) {
        actual = `${entry.bits} bits`;
        ok =
          entry.bits === vector.bits &&
          jsBits === vector.bits &&
          entry.nLo === nLo &&
          entry.nHi === nHi;
      }
      if (ok) gpuPassed++;
      post({
        type: "selftest",
        result: ok ? "PASS" : "FAIL",
        name: `webgpu parity nonce=${vector.nonce}`,
        expected: `${vector.bits} bits`,
        actual,
      });
    } catch (e) {
      post({
        type: "selftest",
        result: "FAIL",
        name: `webgpu parity nonce=${vector.nonce}`,
        expected: `${vector.bits} bits`,
        actual: errText(e),
      });
    }
  }
  verifying = false;

  post({
    type: "gpu-selftest-summary",
    result: gpuPassed === SELF_TEST_VECTORS.length ? "PASS" : "FAIL",
    passed: gpuPassed,
    total: SELF_TEST_VECTORS.length,
    adapter: adapterName,
  });
}

/* ------------------------------------------------------------------ *
 *  Mining loop
 * ------------------------------------------------------------------ */

function randomStartNonce() {
  const words = new Uint32Array(2);
  if (self.crypto && self.crypto.getRandomValues) {
    self.crypto.getRandomValues(words);
  } else {
    words[0] = Date.now() >>> 0;
    words[1] = (Math.random() * 0xffffffff) >>> 0;
  }
  return (BigInt(words[0]) << 32n) | BigInt(words[1]);
}

function nonceHex(value) {
  return "0x" + value.toString(16).padStart(64, "0");
}

function addBest(candidate, found) {
  const list = config.best;
  list.push(candidate);
  list.sort((a, b) => b.bits - a.bits || a.nonce.localeCompare(b.nonce));
  if (list.length > config.keep) list.length = config.keep;
  if (list.includes(candidate)) found.push(candidate);
}

async function start(message) {
  if (running) stopMining("restarted");

  const baseState = buildBase(message.chainId, message.contract, message.miner);
  const startNonce =
    message.startNonce !== undefined && message.startNonce !== null
      ? BigInt(message.startNonce)
      : randomStartNonce();
  const requiredBits = Math.max(0, Number(message.requiredBits ?? 0));
  const threshold = requiredBits > 28 ? requiredBits - NEAR_MARGIN : requiredBits;

  config = {
    scheme: {
      ciBytes: uint256Bytes(message.chainId),
      cBytes: hexToBytes(message.contract, 20),
      mBytes: hexToBytes(message.miner, 20),
    },
    requiredBits,
    threshold,
    keep: Math.max(1, Math.min(Number(message.keep ?? 16), 1000)),
    base: startNonce,
    // Start small: software adapters (SwiftShader) would choke on a big first
    // batch; the adaptive loop doubles iters while cycles stay under ~100 ms.
    iters: MIN_ITERS,
    batch: 1,
    attempts: 0n,
    bestBits: -1,
    best: [],
    startedAt: performance.now(),
    verifyFails: 0,
  };

  try {
    await initGPU(!!message.allowSoftware);
  } catch (e) {
    config = null;
    post({ type: "error", message: errText(e) });
    return;
  }

  // Refuse to mine if the kernel disagrees with the JS reference. The parity
  // vectors use the fixed self-test scheme, so the buffer carries that base
  // until the gate passes; the job base is uploaded right after.
  try {
    gpu.device.queue.writeBuffer(
      gpu.baseBuf,
      0,
      buildBase(SELF_TEST.chainId, SELF_TEST.contract, SELF_TEST.miner),
    );
    const vector = SELF_TEST_VECTORS[2]; // nonce=2138, 15 bits
    const nLo = Number(vector.nonce & 0xffffffffn) >>> 0;
    const nHi = Number((vector.nonce >> 32n) & 0xffffffffn) >>> 0;
    const res = await gpuDispatch(nLo, nHi, 1, 0, 1, 1);
    const entry = res.entries[0];
    const jsBits = verifyNonce(selfTestScheme(), vector.nonce).bits;
    if (!entry || entry.bits !== vector.bits || jsBits !== vector.bits) {
      throw new Error("GPU self-check failed");
    }
  } catch (e) {
    config = null;
    post({ type: "error", message: errText(e) });
    return;
  }

  gpu.device.queue.writeBuffer(gpu.baseBuf, 0, baseState);
  gpu.device.queue.writeBuffer(gpu.candsBuf, 0, new Uint32Array([0]));

  running = true;
  post({
    type: "started",
    startNonce: nonceHex(startNonce),
    adapter: gpu.adapterName,
  });
  loop();
}

async function loop() {
  try {
    while (running) {
      const cycleStart = performance.now();

      const iters = config.iters;

      const { entries, batch: done } = await gpuDispatchBatch(
        iters,
        config.threshold,
        0,
        GRID,
        config.batch,
      );
      if (!running) return;

      const found = [];
      for (const entry of entries) {
        const nonce = (BigInt(entry.nHi) << 32n) | BigInt(entry.nLo);
        const check = verifyNonce(config.scheme, nonce);
        if (check.bits !== entry.bits) {
          // The card must agree with the CPU on every candidate it reports.
          config.verifyFails++;
          if (config.verifyFails >= VERIFY_FAIL_LIMIT) {
            running = false;
            post({
              type: "error",
              message:
                "GPU candidates keep failing local verification — stopping the GPU miner",
            });
            return;
          }
          continue;
        }
        if (check.bits > config.bestBits) config.bestBits = check.bits;
        addBest(
          { nonce: nonceHex(nonce), hash: check.hash, bits: check.bits },
          found,
        );
      }

      config.attempts += done;
      config.base += done;

      const elapsedMs = performance.now() - config.startedAt;
      const hashesPerSecond =
        elapsedMs > 0 ? Number(config.attempts) / (elapsedMs / 1000) : 0;

      post({
        type: "progress",
        attempts: config.attempts.toString(),
        hashesPerSecond,
        bestBits: config.bestBits,
        best: config.best,
        candidates: found,
      });

      // Adaptive batching: keep a cycle near ~120–350 ms so the UI stays live.
      // Grow the per-thread iteration count first, then the number of queued
      // dispatches per sync; shrink in the reverse order (batch first) so the
      // sync cost stays amortized once the GPU is fast.
      const cycleMs = performance.now() - cycleStart;
      if (cycleMs < 120) {
        if (config.iters < MAX_ITERS) {
          config.iters = Math.min(config.iters * 2, MAX_ITERS);
        } else if (config.batch < MAX_DISPATCHES) {
          config.batch *= 2;
        }
      } else if (cycleMs > 350) {
        if (config.batch > 1) {
          config.batch = Math.max(1, Math.floor(config.batch / 2));
        } else if (config.iters > MIN_ITERS) {
          config.iters = Math.max(Math.floor(config.iters / 2), MIN_ITERS);
        }
      }
    }
  } catch (error) {
    running = false;
    post({ type: "error", message: errText(error) });
  }
}

function stopMining(reason = "stopped") {
  if (!running) return;
  running = false;
  post({
    type: "done",
    reason,
    attempts: config ? config.attempts.toString() : "0",
    best: config ? config.best : [],
  });
}

onmessage = ({ data }) => {
  try {
    if (data.type === "start") {
      start(data).catch((error) => {
        running = false;
        post({ type: "error", message: errText(error) });
      });
    } else if (data.type === "stop") {
      stopMining();
    } else if (data.type === "selftest") {
      selfTest(!!data.allowSoftware).catch((error) => {
        verifying = false;
        post({
          type: "gpu-selftest-summary",
          result: "FAIL",
          passed: 0,
          total: SELF_TEST_VECTORS.length,
          adapter: null,
          detail: errText(error),
        });
      });
    }
  } catch (error) {
    running = false;
    post({ type: "error", message: errText(error) });
  }
};
