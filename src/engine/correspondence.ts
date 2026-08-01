// Which dot of the outgoing cloud becomes which dot of the incoming one.
//
// Split out of `blend.ts` because it is decided ONCE per pair of modes, not
// per frame: the answer depends only on the two precomputed clouds, both of
// which are immutable and cached for the life of the app. The blend then
// reads three flat arrays and does no matching work at all.

import type { ModeStaticData } from './types';

/**
 * A resolved pairing. `n` output dots; dot `i` travels from `a[ia[i]]` to
 * `b[ib[i]]`, drawn at `fade[i]`:
 *
 * - `0` — both clouds own this dot; full alpha throughout.
 * - `1` — a duplicate at the `a` end, so it fades IN as it leaves (`m`).
 * - `-1` — a duplicate at the `b` end, so it fades OUT as it lands (`1 - m`).
 *
 * The output always carries `max(nA, nB)` dots so neither pose arrives
 * thinned out. That means the smaller cloud has several output dots sharing
 * one of its dots; they converge on it, and all but the first fade away so
 * the endpoint paints exactly one circle per real dot.
 */
export interface Correspondence {
  n: number;
  ia: Int32Array;
  ib: Int32Array;
  fade: Int8Array;
}

/**
 * Pair every dot of the larger cloud with the nearest dot of the smaller,
 * by position on the sphere.
 *
 * This is the good case, and it is what makes a `searching` → `listening`
 * change read as one lattice easing into another. All three lattice modes
 * are ring-major sphere lattices, but at different ring counts and
 * different dots per ring — 12 rings of up to 29 against 10 of up to 23 —
 * so equal INDICES sit at different latitudes and every dot hops a ring for
 * no reason the eye can attribute to anything. Matching on latitude and
 * longitude instead puts each dot with the one it visually belongs to, and
 * most of them barely move.
 *
 * O(nA·nB) — around 27,000 distance tests for the pair above. That is fine
 * precisely because it happens once per mode pair and is then cached: the
 * clouds are fixed for a given preset, so the answer never changes.
 */
function matchBySphere(
  latA: Float64Array,
  lonA: Float64Array,
  nA: number,
  latB: Float64Array,
  lonB: Float64Array,
  nB: number,
  ia: Int32Array,
  ib: Int32Array
): void {
  // Unit vectors up front: chord distance between them is monotone in the
  // true angle and needs no wrap-around handling, which comparing longitudes
  // directly would (0 and 2π are the same meridian), and it degrades
  // gracefully at the poles, where every longitude names the same point.
  const bx = new Float64Array(nB);
  const by = new Float64Array(nB);
  const bz = new Float64Array(nB);
  for (let j = 0; j < nB; j++) {
    const cl = Math.cos(latB[j]!);
    bx[j] = cl * Math.cos(lonB[j]!);
    by[j] = Math.sin(latB[j]!);
    bz[j] = cl * Math.sin(lonB[j]!);
  }
  for (let i = 0; i < nA; i++) {
    const cl = Math.cos(latA[i]!);
    const x = cl * Math.cos(lonA[i]!);
    const y = Math.sin(latA[i]!);
    const z = cl * Math.sin(lonA[i]!);
    let best = 0;
    let bestD = Infinity;
    for (let j = 0; j < nB; j++) {
      const dx = x - bx[j]!;
      const dy = y - by[j]!;
      const dz = z - bz[j]!;
      const d = dx * dx + dy * dy + dz * dz;
      if (d < bestD) {
        bestD = d;
        best = j;
      }
    }
    ia[i] = i;
    ib[i] = best;
  }
}

/**
 * Resolve the pairing between two built clouds.
 *
 * Uses sphere matching when both modes publish per-dot coordinates (the
 * lattice family), and falls back to proportional index mapping otherwise —
 * `orbits` emits per orbit and `rubik` per lattice cell, so there is no
 * shared frame to match in, and the same FRACTION through each emission
 * order is the best a cheap rule can do. Those pairs take long crossing
 * paths; that is inherent to blending unrelated geometries.
 */
export function buildCorrespondence(
  a: ModeStaticData,
  b: ModeStaticData
): Correspondence {
  const nA = a.dotCount;
  const nB = b.dotCount;
  const n = nA > nB ? nA : nB;
  const ia = new Int32Array(n);
  const ib = new Int32Array(n);
  const fade = new Int8Array(n);

  // Settled orb: both halves resolve to the same cloud. Every dot pairs with
  // itself, so skip the match — for a lattice that would otherwise be a
  // quadratic self-comparison on mount, to derive the identity.
  if (a === b) {
    for (let i = 0; i < n; i++) {
      ia[i] = i;
      ib[i] = i;
    }
    return { n, ia, ib, fade };
  }

  const canMatch =
    a.lat != null && a.lon != null && b.lat != null && b.lon != null;

  if (canMatch) {
    // Always drive the loop from the LARGER cloud, so each of its dots is
    // used exactly once and only the smaller one repeats. Matching the
    // other way round would leave dots of the larger cloud unclaimed, and
    // the pose it belongs to would arrive with holes in it.
    if (nA >= nB) {
      matchBySphere(a.lat!, a.lon!, nA, b.lat!, b.lon!, nB, ia, ib);
    } else {
      matchBySphere(b.lat!, b.lon!, nB, a.lat!, a.lon!, nA, ib, ia);
    }
  } else {
    for (let i = 0; i < n; i++) {
      ia[i] = Math.floor((i * nA) / n);
      ib[i] = Math.floor((i * nB) / n);
    }
  }

  // Mark duplicates. Whichever side is smaller has indices used more than
  // once; the first output dot to claim one keeps full alpha and the rest
  // fade, so the endpoint draws one circle per real dot instead of stacking
  // several on the same spot and darkening it.
  const seen = new Uint8Array(nA > nB ? nA : nB);
  if (nA >= nB) {
    for (let i = 0; i < n; i++) {
      const j = ib[i]!;
      if (seen[j] === 1) fade[i] = -1;
      else seen[j] = 1;
    }
  } else {
    for (let i = 0; i < n; i++) {
      const j = ia[i]!;
      if (seen[j] === 1) fade[i] = 1;
      else seen[j] = 1;
    }
  }

  return { n, ia, ib, fade };
}

/**
 * {@linkcode buildCorrespondence}, memoised on the two cloud objects. They
 * are themselves cached per resolved preset (see `precomputeCached`), so a
 * given pair of states resolves its pairing once for the life of the app —
 * which is what keeps the O(nA·nB) match off the state-change path, where a
 * stall shows up as the orb stuttering before it moves.
 */
const PAIR_CACHE = new WeakMap<
  ModeStaticData,
  WeakMap<ModeStaticData, Correspondence>
>();

export function correspondenceFor(
  a: ModeStaticData,
  b: ModeStaticData
): Correspondence {
  let byB = PAIR_CACHE.get(a);
  if (byB === undefined) {
    byB = new WeakMap();
    PAIR_CACHE.set(a, byB);
  }
  const hit = byB.get(b);
  if (hit !== undefined) return hit;
  const built = buildCorrespondence(a, b);
  byB.set(b, built);
  return built;
}
