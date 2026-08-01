// Ported from thinking-orbs by Jakub Antalik (MIT) — engine/scratch.ts
//
// Reusable per-runtime scratch storage for the render hot path. The web
// original allocated a fresh Dot object (plus a projection tuple) per dot
// per frame; at 60 fps × many orbs that is tens of thousands of short-
// lived objects per second on the UI-thread Hermes heap. Instead, dots
// live in structure-of-arrays Float32Array buffers that are created once
// per JS runtime and reused by every orb: the UI thread runs derived
// values sequentially and each build → record pass completes atomically
// inside one invocation, so a shared buffer can never interleave between
// orbs.

import type { Mat3 } from './core';
import type { ModeDynamics } from './types';

/**
 * Structure-of-arrays dot storage filled by a mode's `build` function
 * each frame and consumed by `recordPicture`.
 *
 * Acquired via {@linkcode acquireDotBuffer}. The buffer is a shared
 * per-runtime scratch: its contents are only valid until the next
 * `acquireDotBuffer` call, so build and paint from it before yielding.
 */
export interface DotBuffer {
  /** Projected x position per dot, in points. */
  xs: Float32Array;
  /** Projected y position per dot, in points. */
  ys: Float32Array;
  /** Depth per dot — painted far→near by ascending z. */
  zs: Float32Array;
  /** Rendered radius per dot, in points (floored at the mode's rMin). */
  rs: Float32Array;
  /** Ink value per dot: 0 = darkest ink on paper. Mirrored on dark themes. */
  ws: Float32Array;
  /** Alpha per dot (1 when the mode does not set one). */
  as: Float32Array;
  /** Number of dots written this frame. Reset by `acquireDotBuffer`. */
  count: number;
  /** Allocated capacity (grows, never shrinks). */
  capacity: number;
}

interface ScratchGlobal {
  __expoThinkingOrbsScratch?: DotBuffer;
  __expoThinkingOrbsScratchB?: DotBuffer;
  __expoThinkingOrbsScratchC?: DotBuffer;
  __expoThinkingOrbsDyn?: ModeDynamics;
  __expoThinkingOrbsMat?: number[];
  __expoThinkingOrbsOrient?: [Float32Array, Float32Array];
}

/**
 * One of two shared `[yaw, pitch]` pairs, for the blend's orientation lock
 * — it reads both modes' orientations for a frame and needs somewhere to
 * put them that costs no allocation. `slot` is 0 or 1; the two are read
 * together, so they cannot share one.
 */
export function acquireOrient(slot: 0 | 1): Float32Array {
  'worklet';
  const g = globalThis as ScratchGlobal;
  let o = g.__expoThinkingOrbsOrient;
  if (o === undefined) {
    o = [new Float32Array(2), new Float32Array(2)];
    g.__expoThinkingOrbsOrient = o;
  }
  return o[slot];
}

/**
 * Return the runtime's shared 3x3 scratch, for a caller building an
 * orientation matrix once per frame. Reused for the same reason as
 * everything else here: a frame should allocate the picture and nothing else.
 *
 * Its contents are only valid until the next call, which is safe for the
 * same reason {@linkcode acquireDotBuffer} is — one build/record pass
 * completes atomically before the runtime yields.
 */
export function acquireMat3(): number[] {
  'worklet';
  const g = globalThis as ScratchGlobal;
  let m = g.__expoThinkingOrbsMat;
  if (m === undefined) {
    m = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    g.__expoThinkingOrbsMat = m;
  }
  return m;
}

/**
 * Return the runtime's shared {@linkcode ModeDynamics} record, with the
 * given per-frame values written into it. Reused rather than allocated so
 * a frame still allocates only the picture itself.
 */
export function acquireDynamics(
  amp: number,
  from: number,
  to: number,
  mix: number,
  yaw: number = 0,
  pitch: number = 0,
  roll: number = 0,
  orient: Mat3 | null = null,
  rMul: number = 1
): ModeDynamics {
  'worklet';
  const g = globalThis as ScratchGlobal;
  let d = g.__expoThinkingOrbsDyn;
  if (d === undefined) {
    d = { amp, from, to, mix, yaw, pitch, roll, orient, rMul };
    g.__expoThinkingOrbsDyn = d;
    return d;
  }
  d.amp = amp;
  d.from = from;
  d.to = to;
  d.mix = mix;
  d.yaw = yaw;
  d.pitch = pitch;
  d.roll = roll;
  d.orient = orient;
  d.rMul = rMul;
  return d;
}

function makeBuffer(capacity: number): DotBuffer {
  'worklet';
  return {
    xs: new Float32Array(capacity),
    ys: new Float32Array(capacity),
    zs: new Float32Array(capacity),
    rs: new Float32Array(capacity),
    ws: new Float32Array(capacity),
    as: new Float32Array(capacity),
    count: 0,
    capacity,
  };
}

/**
 * Return the runtime's shared {@linkcode DotBuffer}, grown to at least
 * `capacity` dots, with `count` reset to 0. Callable from both the React
 * runtime and the Reanimated UI runtime — each keeps its own buffer.
 */
export function acquireDotBuffer(capacity: number): DotBuffer {
  'worklet';
  const g = globalThis as ScratchGlobal;
  let buf = g.__expoThinkingOrbsScratch;
  if (buf === undefined || buf.capacity < capacity) {
    buf = makeBuffer(capacity);
    g.__expoThinkingOrbsScratch = buf;
  }
  buf.count = 0;
  return buf;
}

/**
 * The SECOND shared dot buffer, on its own `globalThis` slot. Exists for
 * one caller: a state blend evaluates two modes for the same frame, so the
 * outgoing pose needs somewhere to live that the incoming build will not
 * overwrite.
 *
 * Same reuse contract as {@linkcode acquireDotBuffer} — valid until the
 * next call, which is safe because a build → blend → record pass completes
 * atomically before the UI runtime yields. Callers that never blend never
 * touch this, so a single-state orb still allocates exactly one buffer per
 * runtime.
 */
export function acquireDotBufferB(capacity: number): DotBuffer {
  'worklet';
  const g = globalThis as ScratchGlobal;
  let buf = g.__expoThinkingOrbsScratchB;
  if (buf === undefined || buf.capacity < capacity) {
    buf = makeBuffer(capacity);
    g.__expoThinkingOrbsScratchB = buf;
  }
  buf.count = 0;
  return buf;
}

/**
 * The THIRD shared dot buffer — where a blend writes its result.
 *
 * It cannot write back over one of its inputs. Position-matched pairing
 * lets an output dot read a source index ABOVE its own, so writing in place
 * would overwrite poses the loop had yet to read, and dots would smear
 * along whatever an earlier iteration happened to leave behind. Ordering
 * the loop cannot fix it: with an arbitrary matching there is no direction
 * that reads only untouched slots.
 */
export function acquireDotBufferC(capacity: number): DotBuffer {
  'worklet';
  const g = globalThis as ScratchGlobal;
  let buf = g.__expoThinkingOrbsScratchC;
  if (buf === undefined || buf.capacity < capacity) {
    buf = makeBuffer(capacity);
    g.__expoThinkingOrbsScratchC = buf;
  }
  buf.count = 0;
  return buf;
}
