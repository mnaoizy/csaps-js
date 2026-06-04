/**
 * csaps-js — Cubic spline approximation (smoothing) for Node.js and the browser.
 *
 * A dependency-free TypeScript port of the Python `csaps` library, supporting
 * univariate, multivariate and N-D gridded data smoothing.
 */

export { csaps, type AutoSmoothingResult, type CsapsOptions } from './csaps';

export {
  CubicSmoothingSpline,
  type CubicSmoothingSplineOptions,
  type EvaluateOptions,
  type UnivariateData,
  type MultivariateData,
} from './umv';

export {
  NdGridCubicSmoothingSpline,
  type NdGridCubicSmoothingSplineOptions,
  type NdEvaluateOptions,
} from './ndg';

export { PPoly, type Extrapolate } from './ppoly';
