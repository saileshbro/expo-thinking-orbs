// Ported from thinking-orbs by Jakub Antalik (MIT) — engine/registry.ts
//
// Mode key → { precompute, build }. Kept separate from the presets so
// tree shaking can in principle drop unused modes in custom builds.

import type { ModeKey } from '../presets';
import {
  buildGlobe,
  buildRubik,
  buildWave,
  orientGlobe,
  orientRubik,
  orientWave,
  precomputeGlobe,
  precomputeRubik,
  precomputeWave,
} from './lattice';
import { buildMorph, precomputeMorph } from './morph';
import { buildOrbits, orientOrbits, precomputeOrbits } from './orbits';
import { buildRibbon, orientRibbon, precomputeRibbon } from './ribbon';
import type { ModeImpl, ModeOpts, ModeStaticData } from './types';
import { buildVoice, precomputeVoice } from './voice';

export const MODES: Record<ModeKey, ModeImpl> = {
  orbits: {
    precompute: precomputeOrbits,
    build: buildOrbits,
    orient: orientOrbits,
  },
  globe: {
    precompute: precomputeGlobe,
    build: buildGlobe,
    orient: orientGlobe,
  },
  rubik: {
    precompute: precomputeRubik,
    build: buildRubik,
    orient: orientRubik,
  },
  wave: { precompute: precomputeWave, build: buildWave, orient: orientWave },
  ribbon: {
    precompute: precomputeRibbon,
    build: buildRibbon,
    orient: orientRibbon,
  },
  // `morph` draws a flat outline and never projects, so it has no
  // orientation to lock. `voice` is one shell blended per dot inside its own
  // builder and never takes the two-mode path at all.
  morph: { precompute: precomputeMorph, build: buildMorph },
  voice: { precompute: precomputeVoice, build: buildVoice },
};

/**
 * Precomputed tables, keyed by the exact options object they were built
 * from. A `WeakMap` because the key is the cached `opts` from
 * `resolvePreset`, which lives as long as the preset does and no longer.
 *
 * Correct because the tables are pure in `opts` and never mutated after
 * `precompute` returns — every consumer, on either thread, only reads them.
 * Each `opts` object is built by `resolvePreset` for exactly one mode, so
 * the object identity already names the mode.
 */
const STATIC_CACHE = new WeakMap<ModeOpts, ModeStaticData>();

/**
 * {@linkcode MODES}`[mode].precompute`, but computed once per resolved
 * preset for the life of the app.
 *
 * This is on the state-change path, which is why it matters. `precompute`
 * is the expensive half of a mode — `buildLattice` grows four plain arrays
 * a dot at a time and copies them into typed arrays — and it runs on the JS
 * thread, inside React's render. A state change needs the tables for BOTH
 * the outgoing and incoming modes, so uncached it stalls the thread twice
 * at exactly the moment the orb is meant to start moving: the tap registers
 * as dropped frames, and the travel only begins once the commit lands.
 *
 * There are six states over two design sizes, and the results are
 * immutable, so nothing is gained by recomputing them. The first visit to a
 * state pays; every later one is a map hit.
 */
export function precomputeCached(
  mode: ModeKey,
  opts: ModeOpts
): ModeStaticData {
  const hit = STATIC_CACHE.get(opts);
  if (hit !== undefined) return hit;
  const data = MODES[mode].precompute(opts);
  STATIC_CACHE.set(opts, data);
  return data;
}
