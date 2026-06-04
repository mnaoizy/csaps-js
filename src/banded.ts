/**
 * Symmetric positive-definite banded linear solver via LDLᵀ factorization.
 *
 * The cubic smoothing spline normal equations produce a symmetric, positive
 * definite, pentadiagonal matrix (half-bandwidth 2). scipy solves this with a
 * sparse solver; here we use a banded LDLᵀ factorization which needs no
 * pivoting for SPD systems and is both fast and numerically stable.
 */

/**
 * Solve `A X = B` where `A` is symmetric positive-definite with half-bandwidth
 * `bw`, supplied in lower-band storage.
 *
 * @param n     Matrix dimension.
 * @param bw    Half-bandwidth (number of sub-diagonals).
 * @param lower Lower-band storage of length `n * (bw + 1)` where
 *              `lower[i * (bw + 1) + o] = A[i][i - o]` for `o = 0..bw`
 *              (entries referencing negative columns are ignored).
 * @param B     Right-hand side, `n` rows × `nrhs` columns (row-major flat array).
 * @param nrhs  Number of right-hand-side columns.
 * @returns     Solution `X`, `n` rows × `nrhs` columns (row-major flat array).
 */
export function ldltBandSolve(
  n: number,
  bw: number,
  lower: Float64Array,
  B: Float64Array,
  nrhs: number,
): Float64Array {
  const stride = bw + 1;
  // L holds the unit-lower-triangular factor in the same band layout
  // (L[i*stride + o] = L[i][i-o], o >= 1); d holds the diagonal of D.
  const L = new Float64Array(n * stride);
  const d = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const oMax = Math.min(bw, i);

    // Off-diagonal factors L[i][i-o], computed for larger o first because the
    // inner sum reads L[i][i-k] with (i-k) > o that must already be finalized.
    for (let o = oMax; o >= 1; o--) {
      const j = i - o;
      let s = lower[i * stride + o]; // A[i][j]
      const kStart = i - bw < 0 ? 0 : i - bw;
      for (let k = kStart; k < j; k++) {
        const oi = i - k;
        const oj = j - k;
        if (oi <= bw && oj <= bw) {
          s -= L[i * stride + oi] * L[j * stride + oj] * d[k];
        }
      }
      L[i * stride + o] = s / d[j];
    }

    // Diagonal D[i]
    let di = lower[i * stride];
    for (let o = 1; o <= oMax; o++) {
      const lio = L[i * stride + o];
      di -= lio * lio * d[i - o];
    }
    if (di === 0 || !Number.isFinite(di)) {
      throw new Error('Banded LDLᵀ solve failed: matrix is singular or ill-conditioned.');
    }
    d[i] = di;
  }

  const X = new Float64Array(n * nrhs);
  X.set(B);

  // Forward substitution: solve L Y = B (in place in X).
  for (let i = 0; i < n; i++) {
    const oMax = Math.min(bw, i);
    for (let c = 0; c < nrhs; c++) {
      let v = X[i * nrhs + c];
      for (let o = 1; o <= oMax; o++) {
        v -= L[i * stride + o] * X[(i - o) * nrhs + c];
      }
      X[i * nrhs + c] = v;
    }
  }

  // Diagonal solve: D Z = Y.
  for (let i = 0; i < n; i++) {
    const inv = 1 / d[i];
    for (let c = 0; c < nrhs; c++) {
      X[i * nrhs + c] *= inv;
    }
  }

  // Back substitution: solve Lᵀ X = Z.
  for (let i = n - 1; i >= 0; i--) {
    const oMaxUp = Math.min(bw, n - 1 - i);
    for (let c = 0; c < nrhs; c++) {
      let v = X[i * nrhs + c];
      for (let o = 1; o <= oMaxUp; o++) {
        v -= L[(i + o) * stride + o] * X[(i + o) * nrhs + c];
      }
      X[i * nrhs + c] = v;
    }
  }

  return X;
}
