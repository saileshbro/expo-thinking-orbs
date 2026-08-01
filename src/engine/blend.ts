// State blending: interpolate two built dot clouds into one pose.
//
// The six ported states are six DIFFERENT mode implementations, not six
// tunings of one — `orbits` emits ring by ring, `rubik` cell by cell, and
// the two do not even agree on how many dots there are. So a state change
// cannot be a parameter animation the way a voice behaviour change is (one
// shell, `buildVoice` lerping two behaviours per dot). It has to run both
// builders for the frame and interpolate the OUTPUTS.
//
// Which dot becomes which is decided once per mode pair, off this path, in
// `correspondence.ts`.

import type { Correspondence } from './correspondence';
import type { DotBuffer } from './scratch';

/**
 * Interpolate `a` and `b` at `m` (0 = `a`'s pose, 1 = `b`'s) into `out`,
 * leaving both inputs untouched. `out` must have capacity for `corr.n`
 * dots.
 *
 * ## This requires an orientation-locked pair
 *
 * A mode's `build` emits PROJECTED screen positions, so what is
 * interpolated here is a picture of a sphere, not a sphere. Straight-line
 * interpolation between two pictures taken at DIFFERENT orientations
 * collapses the shell — `globe` spins at `t*0.5` and `wave` at `t*0.18`, so
 * the same dot sits at opposite screen angles in the two, and the segment
 * joining them cuts through the middle of the orb. Each ring caves in by an
 * amount set by its own phase offset, and the sphere reads as a stack of
 * shrinking tiers before reassembling.
 *
 * The caller is therefore required to draw both halves at ONE common
 * orientation for the frame (see `MODES[].orient` and the lock in
 * `useThinkingOrbPicture`). Given that, matched dots start close together
 * and a straight line between them is the right path. Do not use this on
 * two clouds built at unrelated orientations.
 *
 * ## Fades
 *
 * `corr.fade` marks dots the smaller cloud had to lend out more than once.
 * They converge onto their twin and fade as they arrive, so the endpoint
 * paints exactly one circle per real dot rather than stacking several on
 * one spot and darkening it. Below the painter's 0.02 alpha cutoff they are
 * not even drawn, which is what makes dots appear to split off and merge
 * back rather than to pop.
 *
 * The fade MULTIPLIES the interpolated alpha, never replaces it — `globe`
 * dims its own dots behind the scan meridian via `dimBase`, and overwriting
 * alpha would flatten that for the length of the blend.
 */
export function blendDots(
  out: DotBuffer,
  a: DotBuffer,
  b: DotBuffer,
  m: number,
  corr: Correspondence
): void {
  'worklet';
  const n = corr.n;
  const ias = corr.ia;
  const ibs = corr.ib;
  const fades = corr.fade;

  const axs = a.xs;
  const ays = a.ys;
  const azs = a.zs;
  const ars = a.rs;
  const aws = a.ws;
  const aas = a.as;
  const bxs = b.xs;
  const bys = b.ys;
  const bzs = b.zs;
  const brs = b.rs;
  const bws = b.ws;
  const bas = b.as;
  const oxs = out.xs;
  const oys = out.ys;
  const ozs = out.zs;
  const ors = out.rs;
  const ows = out.ws;
  const oas = out.as;

  for (let i = 0; i < n; i++) {
    const ia = ias[i]!;
    const ib = ibs[i]!;
    const f = fades[i]!;
    // 0 for a dot both clouds own — the common case, and free.
    const fade = f === 0 ? 1 : f === 1 ? m : 1 - m;

    const ax = axs[ia]!;
    const ay = ays[ia]!;
    const az = azs[ia]!;
    const ar = ars[ia]!;
    const aw = aws[ia]!;
    const aa = aas[ia]!;

    oxs[i] = ax + (bxs[ib]! - ax) * m;
    oys[i] = ay + (bys[ib]! - ay) * m;
    ozs[i] = az + (bzs[ib]! - az) * m;
    ors[i] = ar + (brs[ib]! - ar) * m;
    ows[i] = aw + (bws[ib]! - aw) * m;
    oas[i] = (aa + (bas[ib]! - aa) * m) * fade;
  }
  out.count = n;
}
