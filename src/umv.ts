/**
 * Univariate / multivariate cubic smoothing spline.
 *
 * Direct port of the algorithm in `csaps._sspumv` (which itself follows
 * C. de Boor, "A Practical Guide to Splines"). The smoothing spline minimizes
 *
 *     p · Σ wᵢ (yᵢ − f(xᵢ))²  +  (1 − p) · ∫ f''(x)² dx
 *
 * where `p ∈ [0, 1]` is the smoothing parameter: `p = 1` interpolates the data
 * (natural cubic spline), `p = 0` gives the weighted least-squares straight line.
 */

import { ldltBandSolve } from './banded';
import { PPoly, type Extrapolate } from './ppoly';

export interface MakeSplineResult {
  pp: PPoly;
  smooth: number;
}

/** Default smoothing parameter: makes `p·trace(R)` equal `(1−p)·6·trace(QᵀW⁻¹Q)`. */
function computeSmooth(traceR: number, traceQtw: number): number {
  return 1.0 / (1.0 + traceR / (6.0 * traceQtw));
}

/**
 * Normalized smoothing parameter (csaps `normalizedsmooth=True`): rescales `p`
 * so results are invariant to the `x` range and less sensitive to weight and
 * sample-site clumping. See https://github.com/espdev/csaps/pull/47.
 */
function normalizeSmooth(x: Float64Array, w: Float64Array, smooth: number | null): number {
  const n = x.length;
  let xmin = x[0];
  let xmax = x[0];
  for (let i = 1; i < n; i++) {
    if (x[i] < xmin) xmin = x[i];
    if (x[i] > xmax) xmax = x[i];
  }
  const span = xmax - xmin;

  let sumDx2 = 0;
  for (let i = 0; i < n - 1; i++) {
    const d = x[i + 1] - x[i];
    sumDx2 += d * d;
  }
  const effX = 1 + (span * span) / sumDx2;

  let sw = 0;
  let sw2 = 0;
  for (let i = 0; i < n; i++) {
    sw += w[i];
    sw2 += w[i] * w[i];
  }
  const effW = (sw * sw) / sw2;

  const k = 80 * Math.pow(span, 3) * Math.pow(n, -2) * Math.pow(effX, -0.5) * Math.pow(effW, -0.5);

  const s = smooth == null ? 0.5 : smooth;
  return s / (s + (1 - s) * k);
}

/**
 * Build the cubic smoothing spline for `y2d` (an `N × M` array of `N` data
 * components / curves sampled at the `M` sites `x`).
 *
 * @returns a {@link PPoly} with coefficients shaped `(order, pieces, N)` and the
 *          effective smoothing parameter.
 */
export function makeSpline(
  x: Float64Array,
  y2d: number[][],
  w: Float64Array,
  smooth: number | null,
  normalizedsmooth: boolean,
): MakeSplineResult {
  const M = x.length;
  const N = y2d.length;

  const dx = new Float64Array(M - 1);
  for (let i = 0; i < M - 1; i++) {
    dx[i] = x[i + 1] - x[i];
    if (!(dx[i] > 0)) {
      throw new Error("Items of 'xdata' must satisfy x1 < x2 < ... < xN.");
    }
  }

  // Divided differences dy/dx for every component: N × (M-1).
  const dydx: number[][] = new Array(N);
  for (let n = 0; n < N; n++) {
    const row = new Float64Array(M - 1);
    const yn = y2d[n];
    for (let i = 0; i < M - 1; i++) row[i] = (yn[i + 1] - yn[i]) / dx[i];
    dydx[n] = row as unknown as number[];
  }

  // Corner case: two points → linear segment (order-2 spline), p ≡ 1.
  if (M === 2) {
    const c = new Float64Array(2 * 1 * N);
    for (let n = 0; n < N; n++) {
      c[0 * N + n] = dydx[n][0]; // slope
      c[1 * N + n] = y2d[n][0]; // intercept at left break
    }
    return { pp: new PPoly(c, Float64Array.from(x), 2, 1, N), smooth: 1 };
  }

  const m = M - 2;

  const dr = new Float64Array(M - 1); // 1/dx
  for (let i = 0; i < M - 1; i++) dr[i] = 1 / dx[i];
  const iw = new Float64Array(M); // 1/w
  for (let i = 0; i < M; i++) iw[i] = 1 / w[i];

  // Diagonals of QᵀW⁻¹Q (symmetric, pentadiagonal): main, +1, +2.
  const qd = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    const a = dr[i];
    const b = dr[i] + dr[i + 1];
    const cc = dr[i + 1];
    qd[i] = a * a * iw[i] + b * b * iw[i + 1] + cc * cc * iw[i + 2];
  }
  const q1 = new Float64Array(Math.max(0, m - 1));
  for (let i = 0; i < m - 1; i++) {
    q1[i] = -dr[i + 1] * (dr[i] + dr[i + 1]) * iw[i + 1] - dr[i + 1] * (dr[i + 1] + dr[i + 2]) * iw[i + 2];
  }
  const q2 = new Float64Array(Math.max(0, m - 2));
  for (let i = 0; i < m - 2; i++) {
    q2[i] = dr[i + 1] * dr[i + 2] * iw[i + 2];
  }

  // Resolve smoothing parameter.
  let p: number;
  if (normalizedsmooth) {
    p = normalizeSmooth(x, w, smooth);
  } else if (smooth == null) {
    let traceR = 0;
    let traceQ = 0;
    for (let i = 0; i < m; i++) {
      traceR += 2 * (dx[i] + dx[i + 1]);
      traceQ += qd[i];
    }
    p = computeSmooth(traceR, traceQ);
  } else {
    p = smooth;
  }

  const pp6 = 6.0 * (1.0 - p);

  // Symmetric pentadiagonal system A = pp6·QᵀW⁻¹Q + p·R in lower-band storage.
  const lower = new Float64Array(m * 3);
  for (let i = 0; i < m; i++) {
    const rii = 2 * (dx[i] + dx[i + 1]);
    lower[i * 3] = pp6 * qd[i] + p * rii;
    if (i >= 1) lower[i * 3 + 1] = pp6 * q1[i - 1] + p * dx[i]; // A[i,i-1], R[i-1,i] = dx[i]
    if (i >= 2) lower[i * 3 + 2] = pp6 * q2[i - 2]; // A[i,i-2]
  }

  // Right-hand side: second divided difference of dy/dx, shape m × N.
  const B = new Float64Array(m * N);
  for (let i = 0; i < m; i++) {
    for (let n = 0; n < N; n++) B[i * N + n] = dydx[n][i + 1] - dydx[n][i];
  }

  const U = ldltBandSolve(m, 2, lower, B, N); // m × N

  // u padded with zero rows top & bottom → logical index 0..M-1.
  const uAt = (r: number, n: number): number => (r === 0 || r === M - 1 ? 0 : U[(r - 1) * N + n]);

  // d1 = diff(pad(u)) / dx, shape (M-1) × N.
  const d1 = new Float64Array((M - 1) * N);
  for (let i = 0; i < M - 1; i++) {
    const inv = dr[i];
    for (let n = 0; n < N; n++) d1[i * N + n] = (uAt(i + 1, n) - uAt(i, n)) * inv;
  }
  const d1At = (r: number, n: number): number => (r === 0 || r === M ? 0 : d1[(r - 1) * N + n]);

  // d2 = diff(pad(d1)), shape M × N.
  const d2 = new Float64Array(M * N);
  for (let i = 0; i < M; i++) {
    for (let n = 0; n < N; n++) d2[i * N + n] = d1At(i + 1, n) - d1At(i, n);
  }

  // yi = yᵀ − pp6·W⁻¹·d2, shape M × N.
  const yi = new Float64Array(M * N);
  for (let i = 0; i < M; i++) {
    const f = pp6 * iw[i];
    for (let n = 0; n < N; n++) yi[i * N + n] = y2d[n][i] - f * d2[i * N + n];
  }

  // pu = pad(p·u), logical index 0..M-1.
  const puAt = (r: number, n: number): number => (r === 0 || r === M - 1 ? 0 : p * U[(r - 1) * N + n]);

  // Assemble coefficients, shape (4, M-1, N), highest order first.
  const pieces = M - 1;
  const c = new Float64Array(4 * pieces * N);
  for (let i = 0; i < pieces; i++) {
    const h = dx[i];
    const invh = dr[i];
    for (let n = 0; n < N; n++) {
      const pu0 = puAt(i, n);
      const pu1 = puAt(i + 1, n);
      const yi0 = yi[i * N + n];
      const yi1 = yi[(i + 1) * N + n];
      const c1 = (pu1 - pu0) * invh;
      const c2 = 3 * pu0;
      const c3 = (yi1 - yi0) * invh - h * (2 * pu0 + pu1);
      const c4 = yi0;
      c[(0 * pieces + i) * N + n] = c1;
      c[(1 * pieces + i) * N + n] = c2;
      c[(2 * pieces + i) * N + n] = c3;
      c[(3 * pieces + i) * N + n] = c4;
    }
  }

  return { pp: new PPoly(c, Float64Array.from(x), 4, pieces, N), smooth: p };
}

// ---------------------------------------------------------------------------
// Public class
// ---------------------------------------------------------------------------

export type UnivariateData = number[] | Float64Array;
export type MultivariateData = number[] | number[][] | Float64Array;

export interface CubicSmoothingSplineOptions {
  /** Per-site weights (length = number of `x` sites). */
  weights?: UnivariateData;
  /** Smoothing parameter in `[0, 1]`. Omit / `null` to compute it automatically. */
  smooth?: number | null;
  /** Axis of `ydata` that varies with `x` (for 2-D `ydata`). Default `-1` (last). */
  axis?: number;
  /** Normalize the smoothing parameter to be invariant to the `x` range. */
  normalizedsmooth?: boolean;
}

export interface EvaluateOptions {
  /** Derivative order to evaluate (default `0`). */
  nu?: number;
  /** Extrapolate out-of-bounds points (default `true`); otherwise return `NaN`. */
  extrapolate?: Extrapolate;
}

function toFloat64(a: ArrayLike<number>): Float64Array {
  return a instanceof Float64Array ? a : Float64Array.from(a);
}

interface Prepared {
  x: Float64Array;
  y2d: number[][];
  w: Float64Array;
  restore: (flat: Float64Array, L: number) => number[] | number[][];
}

function prepareUnivariate(
  xdata: UnivariateData,
  ydata: MultivariateData,
  weights: UnivariateData | undefined,
  axis: number,
): Prepared {
  const x = toFloat64(xdata);
  const M = x.length;
  if (x.length < 2) throw new Error("'xdata' must contain at least 2 data points.");

  let y2d: number[][];
  let isVector: boolean;
  let transposed = false;

  const first = (ydata as unknown[])[0];
  if (Array.isArray(first) || first instanceof Float64Array) {
    // 2-D ydata: shape [R][C].
    const y = ydata as number[][];
    const R = y.length;
    const C = y[0].length;
    const ndimY = 2;
    const ax = axis < 0 ? ndimY + axis : axis;
    if (ax === 1) {
      if (C !== M) throw new Error(`'ydata' shape[${ax}] (${C}) must equal 'xdata' size (${M}).`);
      y2d = y.map((row) => Array.from(row));
    } else if (ax === 0) {
      if (R !== M) throw new Error(`'ydata' shape[${ax}] (${R}) must equal 'xdata' size (${M}).`);
      // columns are curves → transpose to N × M.
      transposed = true;
      y2d = new Array(C);
      for (let n = 0; n < C; n++) {
        const row = new Array(M);
        for (let i = 0; i < M; i++) row[i] = y[i][n];
        y2d[n] = row;
      }
    } else {
      throw new Error(`Unsupported axis ${axis} for 2-D ydata.`);
    }
    isVector = false;
  } else {
    // 1-D ydata.
    const y = Array.from(ydata as ArrayLike<number>);
    if (y.length !== M) throw new Error(`'ydata' size (${y.length}) must equal 'xdata' size (${M}).`);
    y2d = [y];
    isVector = true;
  }

  let w: Float64Array;
  if (weights == null) {
    w = new Float64Array(M).fill(1);
  } else {
    w = toFloat64(weights);
    if (w.length !== M) throw new Error('Weights vector size must equal xdata size.');
  }

  const N = y2d.length;
  const restore = (flat: Float64Array, L: number): number[] | number[][] => {
    if (isVector) {
      const out = new Array(L);
      for (let q = 0; q < L; q++) out[q] = flat[q];
      return out;
    }
    if (transposed) {
      const out: number[][] = new Array(L);
      for (let q = 0; q < L; q++) {
        const row = new Array(N);
        for (let n = 0; n < N; n++) row[n] = flat[n * L + q];
        out[q] = row;
      }
      return out;
    }
    const out: number[][] = new Array(N);
    for (let n = 0; n < N; n++) {
      const row = new Array(L);
      for (let q = 0; q < L; q++) row[q] = flat[n * L + q];
      out[n] = row;
    }
    return out;
  };

  return { x, y2d, w, restore };
}

/**
 * Cubic smoothing spline for univariate or multivariate data sampled at common
 * sites `x`.
 */
export class CubicSmoothingSpline {
  /** Effective smoothing parameter actually used. */
  readonly smooth: number;
  private readonly pp: PPoly;
  private readonly restore: (flat: Float64Array, L: number) => number[] | number[][];

  constructor(xdata: UnivariateData, ydata: MultivariateData, options: CubicSmoothingSplineOptions = {}) {
    const { weights, smooth = null, axis = -1, normalizedsmooth = false } = options;
    const prep = prepareUnivariate(xdata, ydata, weights, axis);
    const res = makeSpline(prep.x, prep.y2d, prep.w, smooth ?? null, normalizedsmooth);
    this.pp = res.pp;
    this.smooth = res.smooth;
    this.restore = prep.restore;
  }

  /** Evaluate the spline (or one of its derivatives) at the sites `xi`. */
  evaluate(xi: UnivariateData, options: EvaluateOptions = {}): number[] | number[][] {
    const { nu = 0, extrapolate = true } = options;
    const xs = toFloat64(xi);
    const flat = this.pp.evalAll(xs, nu, extrapolate);
    return this.restore(flat, xs.length);
  }

  /** Breakpoints (data sites) of the piecewise polynomial. */
  get breaks(): number[] {
    return Array.from(this.pp.breaks);
  }

  /** Spline description: breakpoints, raw coefficients and shape metadata. */
  get spline(): {
    breaks: number[];
    coeffs: Float64Array;
    order: number;
    pieces: number;
    ndim: number;
  } {
    return {
      breaks: Array.from(this.pp.breaks),
      coeffs: this.pp.c,
      order: this.pp.order,
      pieces: this.pp.pieces,
      ndim: this.pp.ndim,
    };
  }

  /** Internal piecewise polynomial (used by the N-D grid spline). */
  get _pp(): PPoly {
    return this.pp;
  }
}
