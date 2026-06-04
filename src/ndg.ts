/**
 * N-D gridded cubic smoothing spline (piecewise tensor-product polynomial).
 *
 * Port of `csaps._sspndg`. The smoothing spline is built coordinate-wise: the
 * univariate smoother is applied along each grid axis in turn, accumulating a
 * tensor-product coefficient array. Evaluation contracts one axis at a time by
 * reusing the univariate {@link PPoly} evaluator.
 */

import { NdArray, flattenNested, nestArray } from './ndarray';
import { PPoly, type Extrapolate } from './ppoly';
import { makeSpline, type UnivariateData } from './umv';

export interface NdGridCubicSmoothingSplineOptions {
  /** Per-axis weight vectors (each length = that axis' site count). A single vector is allowed for a 1-D grid. */
  weights?: number[][] | number[];
  /** Smoothing parameter(s) in `[0, 1]`: a single value, one per axis, or `null` to auto-compute. */
  smooth?: number | (number | null)[] | null;
  /** Normalize the smoothing parameter to be invariant to each axis' range. */
  normalizedsmooth?: boolean;
}

export interface NdEvaluateOptions {
  /** Per-axis derivative orders (default `0` for every axis). */
  nu?: number[];
  /** Extrapolate out-of-bounds points (default `true`); otherwise return `NaN`. */
  extrapolate?: Extrapolate;
}

function toFloat64(a: ArrayLike<number>): Float64Array {
  return a instanceof Float64Array ? a : Float64Array.from(a);
}

function prepareVectors(data: ArrayLike<number>[], name: string, minSize = 2): Float64Array[] {
  if (!Array.isArray(data)) throw new TypeError(`'${name}' must be a sequence of 1-D vectors.`);
  return data.map((d, axis) => {
    const v = toFloat64(d);
    if (v.length < minSize) {
      throw new Error(`'${name}' must contain at least ${minSize} points for axis ${axis}.`);
    }
    return v;
  });
}

interface BuiltGrid {
  F: NdArray;
  breaks: Float64Array[];
  orders: number[];
  pieces: number[];
  smooths: number[];
}

function buildGrid(
  xs: Float64Array[],
  yNd: NdArray,
  weights: (Float64Array | undefined)[],
  smooth: (number | null)[],
  normalizedsmooth: boolean,
): BuiltGrid {
  const d = xs.length;
  const orders = new Array<number>(d);
  const pieces = new Array<number>(d);
  const smooths = new Array<number>(d);

  // Permutation that moves the last axis to the front.
  const permute = [d - 1];
  for (let a = 0; a < d - 1; a++) permute.push(a);

  let coeffs = yNd;
  let shape = yNd.shape.slice();

  for (let i = d - 1; i >= 0; i--) {
    const lastSize = shape[shape.length - 1];
    const total = coeffs.data.length;
    const Nrest = total / lastSize;

    // Treat the buffer as Nrest "curves" of length lastSize.
    const y2d: number[][] = new Array(Nrest);
    for (let n = 0; n < Nrest; n++) {
      const row = new Array<number>(lastSize);
      const base = n * lastSize;
      for (let col = 0; col < lastSize; col++) row[col] = coeffs.data[base + col];
      y2d[n] = row;
    }

    const w = weights[i] ?? new Float64Array(lastSize).fill(1);
    const res = makeSpline(xs[i], y2d, w, smooth[i], normalizedsmooth);
    const order = res.pp.order;
    const pcs = res.pp.pieces;
    orders[i] = order;
    pieces[i] = pcs;
    smooths[i] = res.smooth;

    // Flatten canonical (order, pcs, Nrest) → (Nrest, order*pcs), order-major in j.
    const op = order * pcs;
    const flat = new Float64Array(Nrest * op);
    for (let k = 0; k < order; k++) {
      for (let ii = 0; ii < pcs; ii++) {
        const j = k * pcs + ii;
        const cBase = j * Nrest;
        for (let n = 0; n < Nrest; n++) flat[n * op + j] = res.pp.c[cBase + n];
      }
    }

    const newShape = shape.slice(0, -1);
    newShape.push(op);
    coeffs = new NdArray(flat, newShape).transpose(permute);
    shape = coeffs.shape.slice();
  }

  return { F: coeffs, breaks: xs, orders, pieces, smooths };
}

function evalGrid(
  built: { F: NdArray; breaks: Float64Array[]; orders: number[]; pieces: number[] },
  xiList: Float64Array[],
  nu: number[],
  extrapolate: Extrapolate,
): NdArray {
  const { F, breaks, orders, pieces } = built;
  const d = breaks.length;

  const permute = [d - 1];
  for (let a = 0; a < d - 1; a++) permute.push(a);

  let coeffs = F;
  let shape = F.shape.slice();

  for (let i = d - 1; i >= 0; i--) {
    const order = orders[i];
    const pcs = pieces[i];
    const lastSize = shape[shape.length - 1]; // order * pcs
    const total = coeffs.data.length;
    const Nrest = total / lastSize;

    // Rebuild canonical (order, pcs, Nrest) from the (Nrest, order*pcs) layout.
    const cc = new Float64Array(order * pcs * Nrest);
    for (let k = 0; k < order; k++) {
      for (let ii = 0; ii < pcs; ii++) {
        const j = k * pcs + ii;
        const dst = (k * pcs + ii) * Nrest;
        for (let n = 0; n < Nrest; n++) cc[dst + n] = coeffs.data[n * lastSize + j];
      }
    }

    const pp = new PPoly(cc, breaks[i], order, pcs, Nrest);
    const xi = xiList[i];
    const Li = xi.length;
    const out = pp.evalAll(xi, nu[i] ?? 0, extrapolate); // (Nrest, Li) row-major

    const newShape = shape.slice(0, -1);
    newShape.push(Li);
    coeffs = new NdArray(out, newShape).transpose(permute);
    shape = coeffs.shape.slice();
  }

  return coeffs;
}

function normalizeSmooth(smooth: number | (number | null)[] | null | undefined, d: number): (number | null)[] {
  if (smooth == null) return new Array(d).fill(null);
  if (Array.isArray(smooth)) {
    if (smooth.length !== d) {
      throw new Error(`Number of smoothing values (${smooth.length}) must equal dimensions (${d}).`);
    }
    return smooth.map((s) => (s == null ? null : s));
  }
  return new Array(d).fill(smooth);
}

function normalizeWeights(
  weights: number[][] | number[] | undefined,
  d: number,
  sizes: number[],
): (Float64Array | undefined)[] {
  if (weights == null) return new Array(d).fill(undefined);
  const w = weights as unknown[];
  const firstIsVector = Array.isArray(w[0]) || (w[0] as unknown) instanceof Float64Array;
  let list: ArrayLike<number>[];
  if (!firstIsVector) {
    // A single flat vector — only valid for a 1-D grid.
    if (d !== 1) throw new Error("'weights' must be a sequence of vectors, one per axis.");
    list = [weights as number[]];
  } else {
    list = weights as number[][];
  }
  if (list.length !== d) throw new Error(`'weights' dimensions (${list.length}) must equal 'xdata' dimensions (${d}).`);
  return list.map((wv, axis) => {
    const v = toFloat64(wv);
    if (v.length !== sizes[axis]) {
      throw new Error(`'weights' size (${v.length}) must equal 'xdata' size (${sizes[axis]}) for axis ${axis}.`);
    }
    return v;
  });
}

/** N-D gridded cubic smoothing spline. */
export class NdGridCubicSmoothingSpline {
  /** Effective per-axis smoothing parameters. */
  readonly smooth: number[];
  private readonly built: BuiltGrid;

  constructor(
    xdata: number[][],
    ydata: number[] | number[][] | number[][][] | unknown,
    options: NdGridCubicSmoothingSplineOptions = {},
  ) {
    const xs = prepareVectors(xdata, 'xdata');
    const d = xs.length;
    const sizes = xs.map((v) => v.length);

    const { data, shape } = flattenNested(ydata);
    if (shape.length !== d) {
      throw new Error(`'ydata' must have ${d} dimensions according to 'xdata' (got ${shape.length}).`);
    }
    for (let a = 0; a < d; a++) {
      if (shape[a] !== sizes[a]) {
        throw new Error(`'ydata' size (${shape[a]}) and 'xdata' size (${sizes[a]}) mismatch for axis ${a}.`);
      }
    }

    const smooth = normalizeSmooth(options.smooth, d);
    const weights = normalizeWeights(options.weights, d, sizes);

    this.built = buildGrid(xs, new NdArray(data, shape), weights, smooth, options.normalizedsmooth ?? false);
    this.smooth = this.built.smooths;
  }

  /** Evaluate the spline on the grid defined by the sequence of site vectors `xi`. */
  evaluate(xi: number[][], options: NdEvaluateOptions = {}): number[] | number[][] | unknown {
    const d = this.built.breaks.length;
    const xiList = prepareVectors(xi, 'xi', 1);
    if (xiList.length !== d) throw new Error(`'xi' must have length ${d} according to the grid.`);
    const nu = options.nu ?? new Array(d).fill(0);
    const extrapolate = options.extrapolate ?? true;
    const result = evalGrid(this.built, xiList, nu, extrapolate);
    return nestArray(result.data, result.shape);
  }

  /** Spline description: per-axis breakpoints, raw coefficient tensor and shape metadata. */
  get spline(): {
    breaks: number[][];
    coeffs: Float64Array;
    coeffsShape: number[];
    order: number[];
    pieces: number[];
  } {
    return {
      breaks: this.built.breaks.map((b) => Array.from(b)),
      coeffs: this.built.F.data,
      coeffsShape: this.built.F.shape.slice(),
      order: this.built.orders.slice(),
      pieces: this.built.pieces.slice(),
    };
  }
}

export type { UnivariateData };
