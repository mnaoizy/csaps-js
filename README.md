# csaps-js

**Cubic spline approximation (smoothing) for Node.js and the browser.**

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mnaoizy/csaps-js/main/assets/surface-dark.svg">
    <img alt="csaps-js 2-D gridded smoothing: a noisy sampled surface smoothed onto a finer grid" src="https://raw.githubusercontent.com/mnaoizy/csaps-js/main/assets/surface-light.svg" width="820">
  </picture>
</p>

<p align="center"><sub>A noisy <code>peaks()</code> surface sampled on a 22×22 grid, smoothed by csaps-js and evaluated on a finer 46×46 grid. <a href="./scripts/make-surface.mjs">(source)</a></sub></p>

A dependency-free TypeScript port of the Python [`csaps`](https://github.com/espdev/csaps)
library. It computes a **cubic smoothing spline** that balances closeness to the
data against smoothness of the curve, for **univariate**, **multivariate** and
**N-D gridded** data.

The smoothing spline `f` minimizes

```
p · Σ wᵢ (yᵢ − f(xᵢ))²  +  (1 − p) · ∫ f''(x)² dx
```

where the smoothing parameter `p ∈ [0, 1]`:

- `p = 0` → the weighted least-squares **straight line**,
- `p = 1` → the natural cubic spline **interpolant** (passes through every point),
- in between → a smooth approximation. If you omit `p`, a sensible value is
  computed automatically from the data.

Results match the reference Python implementation to within floating-point
tolerance (verified by a test suite generated from `csaps` 1.3.3).

## Features

- ✅ Univariate data smoothing (`x: number[]`, `y: number[]`)
- ✅ Multivariate smoothing (many curves sharing the same `x`)
- ✅ N-D gridded data smoothing (2-D surfaces, 3-D volumes, …)
- ✅ Automatic or manual smoothing parameter; optional `normalizedsmooth`
- ✅ Per-site weights
- ✅ Derivative evaluation (`nu`) and optional extrapolation
- ✅ Zero runtime dependencies — works in Node.js, bundlers and the browser
- ✅ ESM + CommonJS + IIFE builds, full TypeScript types

## Installation

```bash
npm install csaps-js
```

## Quick start

```ts
import { csaps } from 'csaps-js';

const x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const y = [2.3, 1.1, 3.8, 3.1, 5.2, 4.9, 7.0, 6.2, 8.1, 9.4];

// Smooth and evaluate at new sites in one call:
const xi = Array.from({ length: 50 }, (_, i) => 1 + (i * 9) / 49);
const yi = csaps(x, y, xi, { smooth: 0.7 }); // number[]

// Let csaps choose the smoothing parameter for you:
const { values, smooth } = csaps(x, y, xi);
console.log('auto smoothing parameter:', smooth);

// Or build a reusable spline object and evaluate later:
const spline = csaps(x, y, { smooth: 0.7 });
const a = spline.evaluate([2.5, 5.5]);     // values
const slope = spline.evaluate([2.5], { nu: 1 }); // first derivative
```

## Usage

### `csaps(x, y, xi?, options?)`

The convenience function. Its return value depends on the arguments:

| Call                              | Returns                                                        |
| --------------------------------- | ------------------------------------------------------------- |
| `csaps(x, y, options?)`           | a spline object (`.evaluate(...)`, `.smooth`, `.spline`)       |
| `csaps(x, y, xi, { smooth })`     | the smoothed values at `xi`                                   |
| `csaps(x, y, xi)` (auto smooth)   | `{ values, smooth }` (an `AutoSmoothingResult`)               |

**Options**

- `smooth` — smoothing parameter in `[0, 1]`. Omit for automatic selection.
  For gridded data this may be a per-axis array (entries may be `null` to
  auto-select that axis).
- `weights` — per-site weights (a vector for univariate data, one vector per
  axis for gridded data).
- `axis` — for 2-D `y`, the axis that varies with `x` (default `-1`, the last).
- `normalizedsmooth` — when `true`, rescales `smooth` so the result is
  invariant to the `x` range and less sensitive to non-uniform spacing/weights.

### Univariate

```ts
import { csaps, CubicSmoothingSpline } from 'csaps-js';

const spline = new CubicSmoothingSpline(x, y, { smooth: 0.85 });
const yi = spline.evaluate(xi);                 // values
const d1 = spline.evaluate(xi, { nu: 1 });      // 1st derivative
const inside = spline.evaluate([-100, 100], { extrapolate: false }); // [NaN, NaN]
console.log(spline.smooth);                      // effective smoothing parameter
```

### Multivariate (many curves, shared `x`)

`y` is a 2-D array of shape `[numCurves][x.length]`; the result has the same
shape `[numCurves][xi.length]`.

```ts
const y2 = [
  [0, 1, 4, 9, 16],   // curve 1
  [0, 1, 8, 27, 64],  // curve 2
];
const yi = csaps([0, 1, 2, 3, 4], y2, [0, 1, 2, 3, 4], { smooth: 0.9 });
// yi: number[][] of shape [2][5]
```

### N-D gridded data

Pass a **list of site vectors** as `x` (one per dimension) and an N-D array as
`y`. Detection is automatic: if `x` is an array of arrays, gridded smoothing is
used.

```ts
import { csaps, NdGridCubicSmoothingSpline } from 'csaps-js';

const xg = [
  [0, 1, 2, 3, 4],       // axis 0 sites
  [0, 1, 2, 3, 4, 5],    // axis 1 sites
];
// z[i][j] = f(xg[0][i], xg[1][j])
const z = xg[0].map((xi) => xg[1].map((yj) => Math.sin(xi) * Math.cos(yj)));

const zi = csaps(xg, z, [
  [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4],
  [0, 1, 2, 3, 4, 5],
], { smooth: 0.9 });
// zi: number[][]  (a smoothed surface on the evaluation grid)

// Per-axis smoothing parameters are also supported:
const surface = new NdGridCubicSmoothingSpline(xg, z, { smooth: [0.5, 0.95] });
console.log(surface.smooth); // [0.5, 0.95]
```

3-D (and higher) grids work the same way: provide three site vectors and a
`number[][][]` for `y`.

## Browser usage

### As a module (bundlers / modern browsers)

```ts
import { csaps } from 'csaps-js';
```

### Via a `<script>` tag (CDN)

The IIFE build exposes a global `csapsjs`:

```html
<script src="https://unpkg.com/csaps-js"></script>
<script>
  const yi = csapsjs.csaps([1, 2, 3, 4], [1, 3, 2, 5], [1, 2, 3, 4], { smooth: 0.8 });
  console.log(yi);
</script>
```

## API surface

- `csaps(x, y, xi?, options?)` — the shortcut function.
- `CubicSmoothingSpline` — univariate / multivariate spline class.
- `NdGridCubicSmoothingSpline` — N-D gridded spline class.
- `PPoly` — the underlying piecewise-polynomial evaluator.
- Types: `CsapsOptions`, `AutoSmoothingResult`, `CubicSmoothingSplineOptions`,
  `NdGridCubicSmoothingSplineOptions`, `EvaluateOptions`, `NdEvaluateOptions`.

Each spline exposes `.smooth` (the effective parameter) and `.spline`
(breakpoints and raw piecewise-polynomial coefficients).

## Notes & differences from Python `csaps`

- Numerics follow de Boor's algorithm exactly; the sparse solve is replaced by
  a banded `LDLᵀ` factorization of the (symmetric, positive-definite,
  pentadiagonal) system.
- The univariate `axis` option supports 1-D `y`, and 2-D `y` with `axis` 0 or
  -1/1. Arbitrary N-D multivariate `y` along an arbitrary axis is not handled
  (use the gridded API or reshape your data).
- Evaluation follows `scipy.interpolate.PPoly` semantics: half-open intervals
  `[x[i], x[i+1])` (last closed), local power basis, first/last-piece
  extrapolation unless `extrapolate: false`.

## License

MIT. This is a port of the MIT-licensed Python
[`csaps`](https://github.com/espdev/csaps) by Eugene Prilepin. See
[LICENSE](./LICENSE).
