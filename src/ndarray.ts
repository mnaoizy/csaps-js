/**
 * Minimal C-contiguous n-dimensional array used by the N-D grid spline.
 *
 * Only the operations required by the tensor-product algorithm are provided:
 * `reshape` (metadata-only, since data is always C-contiguous) and `transpose`
 * (which materializes a permuted copy). This keeps the gridded code free of
 * stride bookkeeping while remaining exact.
 */
export class NdArray {
  constructor(
    readonly data: Float64Array,
    readonly shape: number[],
  ) {}

  get size(): number {
    return this.data.length;
  }

  /** Reinterpret the buffer with a new shape (must keep the same element count). */
  reshape(shape: number[]): NdArray {
    return new NdArray(this.data, shape.slice());
  }

  /** Return a new array with axes permuted according to `perm`, materialized C-contiguous. */
  transpose(perm: number[]): NdArray {
    const d = this.shape.length;
    const newShape = perm.map((axis) => this.shape[axis]);

    const oldStrides = cStrides(this.shape);
    const permutedStrides = perm.map((axis) => oldStrides[axis]);

    const total = this.data.length;
    const out = new Float64Array(total);
    const idx = new Array(d).fill(0);

    for (let pos = 0; pos < total; pos++) {
      let off = 0;
      for (let a = 0; a < d; a++) off += idx[a] * permutedStrides[a];
      out[pos] = this.data[off];
      // Increment the C-order odometer over newShape.
      for (let a = d - 1; a >= 0; a--) {
        if (++idx[a] < newShape[a]) break;
        idx[a] = 0;
      }
    }

    return new NdArray(out, newShape);
  }
}

/** Row-major (C-order) strides for a given shape. */
export function cStrides(shape: number[]): number[] {
  const d = shape.length;
  const strides = new Array(d).fill(1);
  for (let a = d - 2; a >= 0; a--) strides[a] = strides[a + 1] * shape[a + 1];
  return strides;
}

/** Product of a list of numbers (empty list → 1). */
export function prod(xs: number[]): number {
  let p = 1;
  for (const x of xs) p *= x;
  return p;
}

/**
 * Flatten an arbitrarily nested numeric array into a C-contiguous buffer plus
 * its shape. Assumes a rectangular (non-ragged) structure.
 */
export function flattenNested(y: unknown): { data: Float64Array; shape: number[] } {
  const shape: number[] = [];
  let node: unknown = y;
  while (Array.isArray(node)) {
    shape.push(node.length);
    node = node[0];
  }
  const data = new Float64Array(prod(shape));
  let k = 0;
  const fill = (n: unknown, depth: number): void => {
    if (depth === shape.length) {
      data[k++] = n as number;
      return;
    }
    const a = n as unknown[];
    if (a.length !== shape[depth]) {
      throw new Error('flattenNested: input array is ragged (non-rectangular).');
    }
    for (let i = 0; i < a.length; i++) fill(a[i], depth + 1);
  };
  fill(y, 0);
  return { data, shape };
}

/** Rebuild a nested numeric array from a flat C-contiguous buffer and shape. */
export function nestArray(data: ArrayLike<number>, shape: number[]): number[] | number[][] | unknown {
  const strides = cStrides(shape);
  const build = (depth: number, offset: number): unknown => {
    if (depth === shape.length - 1) {
      const row: number[] = new Array(shape[depth]);
      for (let i = 0; i < shape[depth]; i++) row[i] = data[offset + i];
      return row;
    }
    const arr: unknown[] = new Array(shape[depth]);
    for (let i = 0; i < shape[depth]; i++) arr[i] = build(depth + 1, offset + i * strides[depth]);
    return arr;
  };
  return build(0, 0) as number[] | number[][];
}
