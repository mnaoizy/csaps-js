// Generates the 2-D (gridded) demo figure: a noisy sampled surface smoothed by
// csaps-js and rendered as a pair of shaded 3-D surface plots.
//
//   node scripts/make-surface.mjs
//
// Writes assets/surface-light.svg and assets/surface-dark.svg.

import { writeFileSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { csaps } from '../dist/index.js';

const here = dirname(fileURLToPath(import.meta.url));
const outDir = join(here, '..', 'assets');
mkdirSync(outDir, { recursive: true });

// ---------------------------------------------------------------------------
// 1. Noisy 2-D data on a coarse grid, then csaps smoothing on a fine grid
// ---------------------------------------------------------------------------
function lcg(seed) {
  let s = seed >>> 0;
  return () => ((s = (1664525 * s + 1013904223) >>> 0) / 0xffffffff);
}
function gauss(rng) {
  return (rng() + rng() + rng() + rng() - 2) * 0.866;
}
const rng = lcg(424242);

// Classic "peaks" surface.
const peaks = (x, y) =>
  3 * (1 - x) ** 2 * Math.exp(-(x ** 2) - (y + 1) ** 2) -
  10 * (x / 5 - x ** 3 - y ** 5) * Math.exp(-(x ** 2) - y ** 2) -
  (1 / 3) * Math.exp(-((x + 1) ** 2) - y ** 2);

const lin = (a, b, n) => Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));

const NC = 22; // coarse grid
const ax = lin(-3, 3, NC);
const ay = lin(-3, 3, NC);
const Zclean = ax.map((xi) => ay.map((yj) => peaks(xi, yj)));
const Znoisy = Zclean.map((row) => row.map((v) => v + gauss(rng) * 0.6));

const NF = 46; // fine grid for the smoothed surface
const fx = lin(-3, 3, NF);
const fy = lin(-3, 3, NF);
const Zsmooth = csaps([ax, ay], Znoisy, [fx, fy], { smooth: 0.9 });

// Shared height range across both surfaces.
let zmin = Infinity;
let zmax = -Infinity;
for (const row of Znoisy) for (const v of row) (v < zmin && (zmin = v)), v > zmax && (zmax = v);
for (const row of Zsmooth) for (const v of row) (v < zmin && (zmin = v)), v > zmax && (zmax = v);

// ---------------------------------------------------------------------------
// 2. Projection + shading helpers
// ---------------------------------------------------------------------------
const A = -0.62; // azimuth (rad)
const ca = Math.cos(A);
const sa = Math.sin(A);
const SX = 182;
const SY = 100;
const HZ = 156;
const ZK = 1.5; // vertical exaggeration used only for the lighting normals

const norm = (v) => (v - zmin) / (zmax - zmin);

function project(u, v, h, ox, oy) {
  const cu = u - 0.5;
  const cv = v - 0.5;
  const X = cu * ca - cv * sa;
  const depth = cu * sa + cv * ca;
  return [ox + X * SX, oy - depth * SY - h * HZ, depth];
}

// Light direction (upper-left-front), normalized.
const L = (() => {
  const v = [-0.45, -0.55, 0.7];
  const m = Math.hypot(...v);
  return v.map((c) => c / m);
})();

const VIRIDIS = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'];
function hexToRgb(h) {
  const n = parseInt(h.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function viridis(t) {
  const k = Math.max(0, Math.min(1, t)) * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 2, Math.floor(k));
  const f = k - i;
  const a = hexToRgb(VIRIDIS[i]);
  const b = hexToRgb(VIRIDIS[i + 1]);
  return a.map((c, j) => c + (b[j] - c) * f);
}

function quadNormal(p00, p10, p11, p01) {
  // average of the two triangle normals, in world space (cu, cv, h*ZK)
  const e1 = [p10[0] - p00[0], p10[1] - p00[1], p10[2] - p00[2]];
  const e2 = [p01[0] - p00[0], p01[1] - p00[1], p01[2] - p00[2]];
  let n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
  const m = Math.hypot(...n) || 1;
  n = n.map((c) => c / m);
  if (n[2] < 0) n = n.map((c) => -c); // point upward
  return n;
}

// Build the painter-sorted quad list for a surface grid Z (size n×n).
function surfaceQuads(Z, ox, oy) {
  const n = Z.length;
  const quads = [];
  for (let i = 0; i < n - 1; i++) {
    for (let j = 0; j < n - 1; j++) {
      const u0 = i / (n - 1);
      const u1 = (i + 1) / (n - 1);
      const v0 = j / (n - 1);
      const v1 = (j + 1) / (n - 1);
      const h00 = norm(Z[i][j]);
      const h10 = norm(Z[i + 1][j]);
      const h11 = norm(Z[i + 1][j + 1]);
      const h01 = norm(Z[i][j + 1]);
      const s00 = project(u0, v0, h00, ox, oy);
      const s10 = project(u1, v0, h10, ox, oy);
      const s11 = project(u1, v1, h11, ox, oy);
      const s01 = project(u0, v1, h01, ox, oy);

      // world-space corners for the normal
      const w00 = [u0 - 0.5, v0 - 0.5, h00 * ZK];
      const w10 = [u1 - 0.5, v0 - 0.5, h10 * ZK];
      const w11 = [u1 - 0.5, v1 - 0.5, h11 * ZK];
      const w01 = [u0 - 0.5, v1 - 0.5, h01 * ZK];
      const nrm = quadNormal(w00, w10, w11, w01);
      const bright = Math.max(0, nrm[0] * L[0] + nrm[1] * L[1] + nrm[2] * L[2]);
      const shade = 0.45 + 0.6 * bright;

      const hMid = (h00 + h10 + h11 + h01) / 4;
      const base = viridis(hMid);
      const fill = base.map((c) => Math.round(Math.min(255, c * shade)));

      const depth = (s00[2] + s10[2] + s11[2] + s01[2]) / 4;
      quads.push({
        pts: `${s00[0].toFixed(1)},${s00[1].toFixed(1)} ${s10[0].toFixed(1)},${s10[1].toFixed(1)} ${s11[0].toFixed(1)},${s11[1].toFixed(1)} ${s01[0].toFixed(1)},${s01[1].toFixed(1)}`,
        fill: `rgb(${fill[0]},${fill[1]},${fill[2]})`,
        depth,
      });
    }
  }
  quads.sort((p, q) => q.depth - p.depth); // far first
  return quads;
}

// ---------------------------------------------------------------------------
// 3. Render
// ---------------------------------------------------------------------------
const W = 880;
const H = 432;
const FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif";

function render(t) {
  const parts = [];
  const P = (s) => parts.push(s);

  const leftOx = 232;
  const rightOx = 588;
  const oy = 300;

  P(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" font-family="${FONT}" role="img" aria-label="csaps-js 2-D gridded smoothing demo">`);
  P(`<rect width="${W}" height="${H}" rx="14" fill="${t.bg}"/>`);

  // title
  P(`<text x="34" y="38" font-size="22" font-weight="700" fill="${t.title}">csaps-js</text>`);
  P(`<text x="142" y="38" font-size="16" font-weight="500" fill="${t.subtle}">· 2-D gridded smoothing</text>`);

  // surfaces
  for (const q of surfaceQuads(Znoisy, leftOx, oy)) {
    P(`<polygon points="${q.pts}" fill="${q.fill}" stroke="${q.fill}" stroke-width="0.6"/>`);
  }
  for (const q of surfaceQuads(Zsmooth, rightOx, oy)) {
    P(`<polygon points="${q.pts}" fill="${q.fill}" stroke="${q.fill}" stroke-width="0.5"/>`);
  }

  // panel captions
  P(`<text x="${leftOx}" y="${H - 24}" font-size="13.5" font-weight="600" fill="${t.text}" text-anchor="middle">noisy input · 22×22</text>`);
  P(`<text x="${leftOx}" y="${H - 8}" font-size="11" fill="${t.subtle}" text-anchor="middle">peaks() + Gaussian noise</text>`);
  P(`<text x="${rightOx}" y="${H - 24}" font-size="13.5" font-weight="600" fill="${t.text}" text-anchor="middle">csaps smoothed · 46×46</text>`);
  P(`<text x="${rightOx}" y="${H - 8}" font-size="11" fill="${t.subtle}" text-anchor="middle">smooth = 0.9</text>`);

  // colorbar
  const cbX = W - 30;
  const cbY = 66;
  const cbH = 232;
  P(`<defs><linearGradient id="vir" x1="0" y1="1" x2="0" y2="0">`);
  VIRIDIS.forEach((c, i) => P(`<stop offset="${((i / (VIRIDIS.length - 1)) * 100).toFixed(0)}%" stop-color="${c}"/>`));
  P(`</linearGradient></defs>`);
  P(`<rect x="${cbX}" y="${cbY}" width="14" height="${cbH}" rx="3" fill="url(#vir)" stroke="${t.border}" stroke-width="0.5"/>`);
  P(`<text x="${cbX + 7}" y="${cbY - 8}" font-size="11" fill="${t.subtle}" text-anchor="middle">${zmax.toFixed(1)}</text>`);
  P(`<text x="${cbX + 7}" y="${cbY + cbH + 16}" font-size="11" fill="${t.subtle}" text-anchor="middle">${zmin.toFixed(1)}</text>`);
  P(`<text x="${cbX + 7}" y="${cbY + cbH / 2}" font-size="11" fill="${t.subtle}" text-anchor="middle" transform="rotate(90 ${cbX + 7} ${cbY + cbH / 2})" dy="-10">height</text>`);

  P(`</svg>`);
  return parts.join('\n');
}

const themes = {
  light: { bg: '#ffffff', title: '#1f2330', text: '#3c4150', subtle: '#8a90a2', border: '#d8dae2' },
  dark: { bg: '#0d1117', title: '#e8edf4', text: '#c2cad6', subtle: '#7d8696', border: '#30363d' },
};

for (const [name, theme] of Object.entries(themes)) {
  const svg = render(theme);
  writeFileSync(join(outDir, `surface-${name}.svg`), svg);
  console.log(`wrote assets/surface-${name}.svg (${(svg.length / 1024).toFixed(0)} KB)`);
}
