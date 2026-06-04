import { describe, it, expect } from 'vitest';
import ref from './fixtures/reference.json';
import { csaps, CubicSmoothingSpline } from '../src/index';

type Case = Record<string, any>;
const cases = ref as Record<string, Case>;

function flatten(x: any, out: number[] = []): number[] {
  if (Array.isArray(x)) {
    for (const v of x) flatten(v, out);
  } else {
    out.push(x === null ? NaN : (x as number));
  }
  return out;
}

function expectClose(actual: any, expected: any, rtol = 1e-6, atol = 1e-7): void {
  const a = flatten(actual);
  const b = flatten(expected);
  expect(a.length).toBe(b.length);
  let maxErr = 0;
  let worst = -1;
  for (let i = 0; i < a.length; i++) {
    const av = a[i];
    const bv = b[i];
    if (Number.isNaN(bv)) {
      expect(Number.isNaN(av)).toBe(true);
      continue;
    }
    const err = Math.abs(av - bv) - (atol + rtol * Math.abs(bv));
    if (err > maxErr) {
      maxErr = err;
      worst = i;
    }
  }
  if (maxErr > 0) {
    throw new Error(`Mismatch at flat index ${worst}: got ${a[worst]}, expected ${b[worst]} (excess ${maxErr})`);
  }
}

describe('univariate smoothing', () => {
  for (const sm of ['0.0', '0.5', '0.85', '1.0']) {
    it(`matches reference for fixed smooth=${sm}`, () => {
      const c = cases[`uni_fixed_${sm}`];
      const yi = csaps(c.x, c.y, c.xi, { smooth: c.smooth });
      expectClose(yi, c.yi);
    });
  }

  it('smooth=1 interpolates the data exactly', () => {
    const c = cases['uni_fixed_1.0'];
    const yi = csaps(c.x, c.y, c.x, { smooth: 1 }) as number[];
    expectClose(yi, c.y, 1e-9, 1e-9);
  });

  it('auto-computes the smoothing parameter', () => {
    const c = cases['uni_auto'];
    const res = csaps(c.x, c.y, c.xi) as unknown as { values: number[]; smooth: number };
    expect(res.smooth).toBeCloseTo(c.smooth, 9);
    expectClose(res.values, c.yi);
  });

  it('supports per-site weights', () => {
    const c = cases['uni_weights'];
    const yi = csaps(c.x, c.y, c.xi, { smooth: c.smooth, weights: c.weights });
    expectClose(yi, c.yi);
  });

  it('supports normalizedsmooth (fixed)', () => {
    const c = cases['uni_normalized'];
    const sp = csaps(c.x, c.y, { smooth: c.smooth, normalizedsmooth: true }) as CubicSmoothingSpline;
    expect(sp.smooth).toBeCloseTo(c.computed_smooth, 9);
    expectClose(sp.evaluate(c.xi), c.yi);
  });

  it('supports normalizedsmooth (auto)', () => {
    const c = cases['uni_normalized_auto'];
    const sp = csaps(c.x, c.y, { normalizedsmooth: true }) as CubicSmoothingSpline;
    expect(sp.smooth).toBeCloseTo(c.computed_smooth, 9);
    expectClose(sp.evaluate(c.xi), c.yi);
  });
});

describe('derivatives', () => {
  for (const nu of [1, 2, 3]) {
    it(`evaluates derivative nu=${nu}`, () => {
      const c = cases[`uni_deriv_${nu}`];
      const sp = csaps(c.x, c.y, { smooth: c.smooth }) as CubicSmoothingSpline;
      expectClose(sp.evaluate(c.xi, { nu }), c.yi);
    });
  }
});

describe('extrapolation', () => {
  it('extrapolates by default', () => {
    const c = cases['uni_extrap'];
    const sp = csaps(c.x, c.y, { smooth: c.smooth }) as CubicSmoothingSpline;
    expectClose(sp.evaluate(c.xi, { extrapolate: true }), c.yi);
  });

  it('returns NaN outside the domain when extrapolate=false', () => {
    const c = cases['uni_no_extrap'];
    const sp = csaps(c.x, c.y, { smooth: c.smooth }) as CubicSmoothingSpline;
    expectClose(sp.evaluate(c.xi, { extrapolate: false }), c.yi);
  });
});

describe('spline coefficients', () => {
  it('matches reference breaks and coefficients (small case)', () => {
    const c = cases['small_coeffs'];
    const sp = csaps(c.x, c.y, { smooth: c.smooth }) as CubicSmoothingSpline;
    expect(sp.smooth).toBeCloseTo(c.smooth_computed, 9);
    expectClose(sp.breaks, c.breaks, 1e-12, 1e-12);
    // coeffs flat shape (4, pieces, 1) compared against python (4, pieces).
    expectClose(Array.from(sp.spline.coeffs), c.coeffs, 1e-8, 1e-8);
    expectClose(sp.evaluate(c.xi), c.yi);
  });

  it('handles the two-point linear corner case', () => {
    const c = cases['two_point'];
    const sp = csaps(c.x, c.y, { smooth: c.smooth }) as CubicSmoothingSpline;
    expectClose(Array.from(sp.spline.coeffs), c.coeffs, 1e-10, 1e-10);
    expectClose(sp.evaluate(c.xi), c.yi);
  });
});

describe('multivariate smoothing', () => {
  it('smooths multiple curves at once', () => {
    const c = cases['multi'];
    const yi = csaps(c.x, c.y, c.xi, { smooth: c.smooth });
    expectClose(yi, c.yi);
  });

  it('multivariate auto smooth', () => {
    const c = cases['multi_auto'];
    const res = csaps(c.x, c.y, c.xi) as unknown as { values: number[][]; smooth: number };
    expect(res.smooth).toBeCloseTo(c.smooth, 9);
    expectClose(res.values, c.yi);
  });
});

describe('N-D gridded smoothing', () => {
  it('2-D grid, fixed smooth', () => {
    const c = cases['ndgrid_2d'];
    const yi = csaps(c.x, c.y, c.xi, { smooth: c.smooth });
    expectClose(yi, c.yi);
  });

  it('2-D grid, auto smooth', () => {
    const c = cases['ndgrid_2d_auto'];
    const res = csaps(c.x, c.y, c.xi) as unknown as { values: number[][]; smooth: number[] };
    for (let a = 0; a < c.smooth.length; a++) expect(res.smooth[a]).toBeCloseTo(c.smooth[a], 9);
    expectClose(res.values, c.yi);
  });

  it('2-D grid, per-axis smooth', () => {
    const c = cases['ndgrid_2d_persmooth'];
    const yi = csaps(c.x, c.y, c.xi, { smooth: c.smooth });
    expectClose(yi, c.yi);
  });

  it('1-D grid behaves like univariate', () => {
    const c = cases['ndgrid_1d'];
    const yi = csaps(c.x, c.y, c.xi, { smooth: c.smooth });
    expectClose(yi, c.yi);
  });

  it('3-D grid, fixed smooth', () => {
    const c = cases['ndgrid_3d'];
    const yi = csaps(c.x, c.y, c.xi, { smooth: c.smooth });
    expectClose(yi, c.yi);
  });
});
