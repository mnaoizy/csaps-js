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

  /** Evaluate component `n` at scalar `xq` with derivative order `nu`. */
  private evalScalar(n: number, xq: number, nu: number, extrapolate: Extrapolate): number {
    const i = this.findPiece(xq, extrapolate);
    if (i < 0) return NaN;
    const s = xq - this.breaks[i];
    const order = this.order;
    const pieces = this.pieces;
    const ndim = this.ndim;
    let res = 0;
    for (let k = 0; k < order; k++) {
      const e = order - 1 - k; // power of s for this coefficient
      if (e < nu) continue;
      let f = 1;
      for (let t = 0; t < nu; t++) f *= e - t; // falling factorial e·(e-1)···(e-nu+1)
      const coef = this.c[(k * pieces + i) * ndim + n];
      res += coef * f * Math.pow(s, e - nu);
    }
    return res;
  }

  /**
   * Evaluate all components at every point in `xs`.
   *
   * @returns Flat row-major array with logical shape `(ndim, xs.length)`:
   *          `out[n * xs.length + q]`.
   */
  evalAll(xs: ArrayLike<number>, nu: number, extrapolate: Extrapolate): Float64Array {
    const L = xs.length;
    const out = new Float64Array(this.ndim * L);
    for (let q = 0; q < L; q++) {
      const xq = xs[q];
      for (let n = 0; n < this.ndim; n++) {
        out[n * L + q] = this.evalScalar(n, xq, nu, extrapolate);
      }
    }
    return out;
  }
}
