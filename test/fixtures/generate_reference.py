"""Generate reference values from the Python csaps library for the TS port tests."""
import json
import numpy as np
from csaps import csaps, CubicSmoothingSpline, NdGridCubicSmoothingSpline

np.random.seed(1234)


def arr(a):
    return np.asarray(a, dtype=float).tolist()


cases = {}

# ---- 1. Univariate, fixed smooth, evaluate ----
x = np.linspace(-5.0, 5.0, 25)
y = np.exp(-((x / 2.5) ** 2)) + (np.sin(x) * 0.3)
xi = np.linspace(-5.0, 5.0, 60)
for sm in (0.0, 0.5, 0.85, 1.0):
    yi = csaps(x, y, xi, smooth=sm)
    cases[f"uni_fixed_{sm}"] = {
        "x": arr(x), "y": arr(y), "xi": arr(xi), "smooth": sm, "yi": arr(yi),
    }

# ---- 2. Univariate auto smooth ----
res = csaps(x, y, xi)
cases["uni_auto"] = {
    "x": arr(x), "y": arr(y), "xi": arr(xi),
    "yi": arr(res.values), "smooth": float(res.smooth),
}

# ---- 3. Univariate with weights ----
w = np.linspace(0.5, 2.0, x.size)
yi = csaps(x, y, xi, weights=w, smooth=0.8)
cases["uni_weights"] = {
    "x": arr(x), "y": arr(y), "xi": arr(xi), "weights": arr(w),
    "smooth": 0.8, "yi": arr(yi),
}

# ---- 4. normalizedsmooth ----
sp = csaps(x, y, smooth=0.5, normalizedsmooth=True)
yi = sp(xi)
cases["uni_normalized"] = {
    "x": arr(x), "y": arr(y), "xi": arr(xi), "smooth": 0.5,
    "normalizedsmooth": True, "yi": arr(yi), "computed_smooth": float(sp.smooth),
}

# ---- 5. normalizedsmooth auto (smooth=None) ----
sp = csaps(x, y, normalizedsmooth=True)
yi = sp(xi)
cases["uni_normalized_auto"] = {
    "x": arr(x), "y": arr(y), "xi": arr(xi),
    "normalizedsmooth": True, "yi": arr(yi), "computed_smooth": float(sp.smooth),
}

# ---- 6. Derivatives ----
sp = csaps(x, y, smooth=0.85)
for nu in (1, 2, 3):
    cases[f"uni_deriv_{nu}"] = {
        "x": arr(x), "y": arr(y), "xi": arr(xi), "smooth": 0.85,
        "nu": nu, "yi": arr(sp(xi, nu=nu)),
    }

# ---- 7. Extrapolation off ----
xi_ext = np.linspace(-7.0, 7.0, 40)
sp = csaps(x, y, smooth=0.85)
yi = sp(xi_ext, extrapolate=False)
cases["uni_no_extrap"] = {
    "x": arr(x), "y": arr(y), "xi": arr(xi_ext), "smooth": 0.85,
    "extrapolate": False,
    "yi": [None if not np.isfinite(v) else float(v) for v in yi],
}
# extrapolation on
yi = sp(xi_ext, extrapolate=True)
cases["uni_extrap"] = {
    "x": arr(x), "y": arr(y), "xi": arr(xi_ext), "smooth": 0.85,
    "extrapolate": True, "yi": arr(yi),
}

# ---- 8. Small case: exact coefficients ----
xs = [1.0, 2.0, 4.0, 5.0, 7.0]
ys = [1.0, 3.0, 2.0, 5.0, 4.0]
sp = csaps(xs, ys, smooth=0.7)
cases["small_coeffs"] = {
    "x": arr(xs), "y": arr(ys), "smooth": 0.7,
    "smooth_computed": float(sp.smooth),
    "breaks": arr(sp.spline.breaks),
    "coeffs": arr(sp.spline.coeffs),  # shape (4, pieces)
    "xi": arr(np.linspace(1, 7, 31)),
    "yi": arr(sp(np.linspace(1, 7, 31))),
}

# two-point linear corner case
sp = csaps([0.0, 2.0], [1.0, 5.0], smooth=0.9)
cases["two_point"] = {
    "x": [0.0, 2.0], "y": [1.0, 5.0], "smooth": 0.9,
    "breaks": arr(sp.spline.breaks), "coeffs": arr(sp.spline.coeffs),
    "xi": arr(np.linspace(-1, 3, 9)), "yi": arr(sp(np.linspace(-1, 3, 9))),
}

# ---- 9. Multivariate (2 curves, shape (2, M)) ----
ym = np.vstack([y, np.cos(x) * 2.0])
xi_m = np.linspace(-5, 5, 40)
yi = csaps(x, ym, xi_m, smooth=0.8)
cases["multi"] = {
    "x": arr(x), "y": arr(ym), "xi": arr(xi_m), "smooth": 0.8, "yi": arr(yi),
}

# multivariate auto
res = csaps(x, ym, xi_m)
cases["multi_auto"] = {
    "x": arr(x), "y": arr(ym), "xi": arr(xi_m),
    "yi": arr(res.values), "smooth": float(res.smooth),
}

# ---- 10. N-D grid 2-D ----
x2 = [np.linspace(0.0, 5.0, 11), np.linspace(0.0, 6.0, 13)]
xx, yy = np.meshgrid(*x2, indexing="ij")
z = np.sin(xx) * np.cos(yy) + (np.random.rand(11, 13) - 0.5) * 0.2
xi2 = [np.linspace(0.0, 5.0, 23), np.linspace(0.0, 6.0, 27)]
zi = csaps(x2, z, xi2, smooth=0.9)
cases["ndgrid_2d"] = {
    "x": [arr(a) for a in x2], "y": arr(z),
    "xi": [arr(a) for a in xi2], "smooth": 0.9, "yi": arr(zi),
}

# N-D grid auto
res = csaps(x2, z, xi2)
cases["ndgrid_2d_auto"] = {
    "x": [arr(a) for a in x2], "y": arr(z), "xi": [arr(a) for a in xi2],
    "yi": arr(res.values), "smooth": [float(s) for s in res.smooth],
}

# N-D grid per-axis smooth
zi = csaps(x2, z, xi2, smooth=[0.5, 0.95])
cases["ndgrid_2d_persmooth"] = {
    "x": [arr(a) for a in x2], "y": arr(z), "xi": [arr(a) for a in xi2],
    "smooth": [0.5, 0.95], "yi": arr(zi),
}

# ---- 11. N-D grid 1-D (single axis) behaves like univariate ----
x1 = [np.linspace(0.0, 10.0, 15)]
yv = np.sin(x1[0])
xi1 = [np.linspace(0.0, 10.0, 30)]
zi = csaps(x1, yv, xi1, smooth=0.8)
cases["ndgrid_1d"] = {
    "x": [arr(a) for a in x1], "y": arr(yv), "xi": [arr(a) for a in xi1],
    "smooth": 0.8, "yi": arr(zi),
}

# ---- 12. N-D grid 3-D ----
x3 = [np.linspace(0, 1, 6), np.linspace(0, 1, 7), np.linspace(0, 1, 5)]
g = np.meshgrid(*x3, indexing="ij")
v = np.sin(g[0] * 3) + g[1] ** 2 - g[2]
xi3 = [np.linspace(0, 1, 9), np.linspace(0, 1, 8), np.linspace(0, 1, 7)]
vi = csaps(x3, v, xi3, smooth=0.85)
cases["ndgrid_3d"] = {
    "x": [arr(a) for a in x3], "y": arr(v), "xi": [arr(a) for a in xi3],
    "smooth": 0.85, "yi": arr(vi),
}

with open("/Users/mnaoizy/csaps-js/test/fixtures/reference.json", "w") as f:
    json.dump(cases, f)

print("wrote", len(cases), "cases")
for k, val in cases.items():
    sm = val.get("smooth", val.get("computed_smooth", val.get("smooth_computed")))
    print(f"  {k}: smooth={sm}")
