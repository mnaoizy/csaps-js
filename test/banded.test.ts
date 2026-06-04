import { describe, it, expect } from 'vitest';
import { ldltBandSolve } from '../src/banded';

/** Deterministic LCG so the test is reproducible without external deps. */
function makeRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
}

/**
 * Build a random symmetric positive-definite pentadiagonal matrix (half-bw 2)
 * in both dense and lower-band form.
 */
function makePentaSPD(n: number, rng: () => number): { dense: number[][]; lower: Float64Array } {
  const A: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let o = 1; o <= 2; o++) {
      if (i - o >= 0) {
        const v = rng() - 0.5;
        A[i][i - o] = v;
        A[i - o][i] = v;
      }
    }
  }
  // Make strictly diagonally dominant → SPD.
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) if (j !== i) sum += Math.abs(A[i][j]);
    A[i][i] = sum + 1 + rng();
  }
  const lower = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    lower[i * 3] = A[i][i];
    if (i >= 1) lower[i * 3 + 1] = A[i][i - 1];
    if (i >= 2) lower[i * 3 + 2] = A[i][i - 2];
  }
  return { dense: A, lower };
}

describe('ldltBandSolve', () => {
  it('solves A X = B with small residual (single RHS)', () => {
    const rng = makeRng(42);
    const n = 50;
    const { dense, lower } = makePentaSPD(n, rng);
    const B = new Float64Array(n);
    for (let i = 0; i < n; i++) B[i] = rng() * 10 - 5;

    const X = ldltBandSolve(n, 2, lower, B, 1);

    // residual = A X - B
    let maxRes = 0;
    for (let i = 0; i < n; i++) {
      let v = 0;
      for (let j = 0; j < n; j++) v += dense[i][j] * X[j];
      maxRes = Math.max(maxRes, Math.abs(v - B[i]));
    }
    expect(maxRes).toBeLessThan(1e-9);
  });

  it('solves A X = B for multiple RHS columns', () => {
    const rng = makeRng(7);
    const n = 30;
    const nrhs = 4;
    const { dense, lower } = makePentaSPD(n, rng);
    const B = new Float64Array(n * nrhs);
    for (let i = 0; i < n * nrhs; i++) B[i] = rng() * 2 - 1;

    const X = ldltBandSolve(n, 2, lower, B, nrhs);

    let maxRes = 0;
    for (let c = 0; c < nrhs; c++) {
      for (let i = 0; i < n; i++) {
        let v = 0;
        for (let j = 0; j < n; j++) v += dense[i][j] * X[j * nrhs + c];
        maxRes = Math.max(maxRes, Math.abs(v - B[i * nrhs + c]));
      }
    }
    expect(maxRes).toBeLessThan(1e-9);
  });

  it('handles a 1x1 system', () => {
    const lower = new Float64Array([4]);
    const B = new Float64Array([8]);
    const X = ldltBandSolve(1, 2, lower, B, 1);
    expect(X[0]).toBeCloseTo(2, 12);
  });
});
