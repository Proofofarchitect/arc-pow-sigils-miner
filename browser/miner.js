"use strict";

export class PowMiner {
  constructor(workerUrl = "./miner-worker.js") {
    this.worker = new Worker(workerUrl);
    this.running = false;

    this.onStarted = null;
    this.onProgress = null;
    this.onCandidate = null;
    this.onDone = null;
    this.onError = null;
    this.onSelfTest = null;
    this.onSelfTestSummary = null;

    this.worker.onmessage = ({ data }) => this.#handle(data);
    this.worker.onerror = (event) => {
      this.running = false;
      this.onError?.({ type: "error", message: event.message });
    };
  }

  start({
    chainId,
    contract,
    miner,
    startNonce = 0,
    keep = 16,
    batchSize = 4096
  }) {
    this.worker.postMessage({
      type: "start",
      chainId,
      contract,
      miner,
      startNonce,
      keep,
      batchSize
    });
  }

  stop() {
    this.worker.postMessage({ type: "stop" });
  }

  selfTest() {
    this.worker.postMessage({ type: "selftest" });
  }

  terminate() {
    this.running = false;
    this.worker.terminate();
  }

  #handle(data) {
    switch (data.type) {
      case "started":
        this.running = true;
        this.onStarted?.(data);
        break;

      case "progress":
        for (const candidate of data.candidates || []) {
          this.onCandidate?.(candidate, data);
        }
        this.onProgress?.(data);
        break;

      case "done":
        this.running = false;
        this.onDone?.(data);
        break;

      case "error":
        this.running = false;
        this.onError?.(data);
        break;

      case "selftest":
        this.onSelfTest?.(data);
        break;

      case "selftest-summary":
        this.onSelfTestSummary?.(data);
        break;
    }
  }
}

export function formatRate(hashesPerSecond) {
  if (hashesPerSecond >= 1e6) return `${(hashesPerSecond / 1e6).toFixed(2)} MH/s`;
  if (hashesPerSecond >= 1e3) return `${(hashesPerSecond / 1e3).toFixed(2)} kH/s`;
  return `${Math.round(hashesPerSecond)} H/s`;
}

export function formatAttempts(attempts) {
  return BigInt(attempts).toLocaleString("en-US");
}
