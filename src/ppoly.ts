/**
 * Piecewise-polynomial evaluation in local power basis.
 *
 * Mirrors the relevant behaviour of `scipy.interpolate.PPoly`:
 *   - coefficients are stored highest-order first,
 *   - each piece is evaluated in the local variable `s = xq - breaks[i]`,
 *   - intervals are half-open `[x[i], x[i+1])` except the last, which is closed,
 *   - out-of-bounds points are extrapolated from the first/last piece (or NaN).
 */

export type Extrapolate = boolean;

/**
 * Univariate (optionally multi-component) piecewise polynomial.
 *
 * Coefficients are stored as a flat C-contiguous array with logical shape
 * `(order, pieces, ndim)`: `c[k][i][n] = data[(k * pieces + i) * ndim + n]`,
 * where `k` indexes the polynomial coefficient (highest power first), `i` the
 * piece and `n` the data component (`ndim === 1` for univariate data).
 */
export class PPoly {
  constructor(
    readonly c: Float64Array,
    readonly breaks: Float64Array,
    readonly order: number,
    readonly pieces: number,
    readonly ndim: number,
  ) {}

  /** Locate the piece index for `xq`, or `-1` when out of bounds and not extrapolating. */
  private findPiece(xq: number, extrapolate: Extrapolate): number {
    const b = this.breaks;
    const last = this.pieces;
    if (xq < b[0]) return extrapolate ? 0 : -1;
    if (xq > b[last]) return extrapolate ? this.pieces - 1 : -1;
    if (xq >= b[last]) return this.pieces - 1;
    // Binary search for the largest i with breaks[i] <= xq.
    let lo = 0;
    let hi = this.pieces - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (b[mid] <= xq) lo = mid;
      else hi = mid - 1;
    }
    return lo;
  }

  /**
   * Evaluate all components at every point in `xs` (Horner's method).
   *
   * The piece lookup is done once per evaluation point and reused across all
   * components, and the polynomial is evaluated with Horner's scheme rather
   * than `Math.pow` — both matter for multivariate and N-D gridded data where
   * `ndim` is large.
   *
   * @returns Flat row-major array with logical shape `(ndim, xs.length)`:
   *          `out[n * xs.length + q]`.
   */
  evalAll(xs: ArrayLike<number>, nu: number, extrapolate: Extrapolate): Float64Array {
    const { order, pieces, ndim, breaks, c } = this;
    const deg = order - 1;
    const L = xs.length;
    const out = new Float64Array(ndim * L);
    if (nu > deg) return out; // derivative of order > degree is identically zero

    const effDeg = deg - nu;
    const kstep = pieces * ndim;

    // Per-coefficient derivative multipliers (depend only on deg, nu, k).
    let mult: Float64Array | null = null;
    if (nu !== 0) {
      mult = new Float64Array(effDeg + 1);
      for (let k = 0; k <= effDeg; k++) mult[k] = fallingFactorial(deg - k, nu);
    }

    for (let q = 0; q < L; q++) {
      const xq = xs[q];
      const i = this.findPiece(xq, extrapolate);
      if (i < 0) {
        for (let n = 0; n < ndim; n++) out[n * L + q] = NaN;
        continue;
      }
      const s = xq - breaks[i];
      const start = i * ndim;

      if (nu === 0) {
        // Value: plain Horner over the highest-first coefficients.
        for (let n = 0; n < ndim; n++) {
          let off = start + n;
          let res = c[off];
          for (let k = 1; k <= deg; k++) {
            off += kstep;
            res = res * s + c[off];
          }
          out[n * L + q] = res;
        }
      } else {
        // Derivative: Horner over the differentiated coefficients
        // c[k] · (deg-k)·(deg-k-1)···(deg-k-nu+1), for k = 0..effDeg.
        const m = mult!;
        for (let n = 0; n < ndim; n++) {
          let off = start + n;
          let res = c[off] * m[0];
          for (let k = 1; k <= effDeg; k++) {
            off += kstep;
            res = res * s + c[off] * m[k];
          }
          out[n * L + q] = res;
        }
      }
    }
    return out;
  }
}

/** Falling factorial base·(base-1)···(base-nu+1). */
function fallingFactorial(base: number, nu: number): number {
  let f = 1;
  for (let t = 0; t < nu; t++) f *= base - t;
  return f;
}
