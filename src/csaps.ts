/**
 * The `csaps` shortcut function — the primary entry point for smoothing data.
 */

import {
  CubicSmoothingSpline,
  type CubicSmoothingSplineOptions,
  type MultivariateData,
  type UnivariateData,
} from './umv';
import { NdGridCubicSmoothingSpline, type NdGridCubicSmoothingSplineOptions } from './ndg';

/** Result returned when smoothing data while the smoothing parameter is computed automatically. */
export interface AutoSmoothingResult {
  /** Smoothed data values. */
  values: number[] | number[][] | unknown;
  /** The smoothing parameter(s) that were computed. */
  smooth: number | number[];
}

export interface CsapsOptions {
  /** Weights: a single vector (univariate) or one vector per axis (gridded). */
  weights?: number[] | number[][] | Float64Array;
  /** Smoothing parameter(s) in `[0, 1]`: a value, one per grid axis, or `null`/omitted to auto-compute. */
  smooth?: number | (number | null)[] | null;
  /** Axis of `ydata` that varies with `x` (univariate 2-D data only). Default `-1`. */
  axis?: number;
  /** Normalize the smoothing parameter to be invariant to the data range. */
  normalizedsmooth?: boolean;
}

function isVectorList(x: unknown): boolean {
  return Array.isArray(x) && x.length > 0 && (Array.isArray(x[0]) || x[0] instanceof Float64Array);
}

function isArrayLike(x: unknown): boolean {
  return Array.isArray(x) || x instanceof Float64Array;
}

// --- Overloads -------------------------------------------------------------

/** Build a univariate / multivariate smoothing spline (no evaluation sites given). */
export function csaps(
  xdata: UnivariateData,
  ydata: MultivariateData,
  options?: CubicSmoothingSplineOptions,
): CubicSmoothingSpline;
/** Build an N-D gridded smoothing spline (no evaluation sites given). */
export function csaps(
  xdata: number[][],
  ydata: number[] | number[][] | number[][][],
  options?: NdGridCubicSmoothingSplineOptions,
): NdGridCubicSmoothingSpline;
/** Smooth univariate / multivariate data at the given evaluation sites. */
export function csaps(
  xdata: UnivariateData,
  ydata: MultivariateData,
  xidata: UnivariateData,
  options?: CubicSmoothingSplineOptions,
): number[] | number[][] | AutoSmoothingResult;
/** Smooth N-D gridded data at the given evaluation grid. */
export function csaps(
  xdata: number[][],
  ydata: number[] | number[][] | number[][][],
  xidata: number[][],
  options?: NdGridCubicSmoothingSplineOptions,
): number[] | number[][] | unknown | AutoSmoothingResult;

// --- Implementation --------------------------------------------------------

export function csaps(
  xdata: UnivariateData | number[][],
  ydata: unknown,
  arg3?: unknown,
  arg4?: unknown,
):
  | CubicSmoothingSpline
  | NdGridCubicSmoothingSpline
  | number[]
  | number[][]
  | unknown
  | AutoSmoothingResult {
  let xidata: unknown;
  let options: CsapsOptions;

  if (arg3 === undefined) {
    xidata = undefined;
    options = (arg4 as CsapsOptions) ?? {};
  } else if (isArrayLike(arg3)) {
    xidata = arg3;
    options = (arg4 as CsapsOptions) ?? {};
  } else {
    xidata = undefined;
    options = (arg3 as CsapsOptions) ?? {};
  }

  const grid = isVectorList(xdata);

  const spline: CubicSmoothingSpline | NdGridCubicSmoothingSpline = grid
    ? new NdGridCubicSmoothingSpline(xdata as number[][], ydata, {
        weights: options.weights as number[][] | number[] | undefined,
        smooth: options.smooth,
        normalizedsmooth: options.normalizedsmooth,
      })
    : new CubicSmoothingSpline(xdata as UnivariateData, ydata as MultivariateData, {
        weights: options.weights as UnivariateData | undefined,
        smooth: typeof options.smooth === 'number' ? options.smooth : (options.smooth as null | undefined) ?? null,
        axis: options.axis,
        normalizedsmooth: options.normalizedsmooth,
      });

  if (xidata === undefined) return spline;

  const values = grid
    ? (spline as NdGridCubicSmoothingSpline).evaluate(xidata as number[][])
    : (spline as CubicSmoothingSpline).evaluate(xidata as UnivariateData);

  const sm = options.smooth;
  const autoSmooth = Array.isArray(sm) ? sm.some((s) => s == null) : sm == null;

  if (autoSmooth) {
    return { values, smooth: spline.smooth } as AutoSmoothingResult;
  }
  return values as number[] | number[][];
}
