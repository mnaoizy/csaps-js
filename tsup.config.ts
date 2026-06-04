import { defineConfig } from 'tsup';

export default defineConfig({
  entry: ['src/index.ts'],
  format: ['esm', 'cjs', 'iife'],
  globalName: 'csapsjs',
  dts: true,
  clean: true,
  sourcemap: true,
  minify: false,
  treeshake: true,
  target: 'es2020',
  outExtension({ format }) {
    if (format === 'cjs') return { js: '.cjs' };
    if (format === 'iife') return { js: '.global.js' };
    return { js: '.js' };
  },
});
