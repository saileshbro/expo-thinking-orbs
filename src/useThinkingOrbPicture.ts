// Ported from thinking-orbs by Jakub Antalik (MIT) — useThinkingOrbPicture.ts
//
// The orb render loop as a hook: clock, engine build and Skia picture
// recording, without the component's <Canvas>. `ThinkingOrb` is a thin
// wrapper over this; consumers with several orbs (or orbs plus other
// animated Skia content) can embed the returned picture in ONE shared
// <Canvas> instead of mounting one Skia view per orb — on Android every
// Skia view is a separate hardware-buffer surface composited per frame,
// so fewer, larger canvases render dramatically cheaper.

import { useEffect, useMemo, useRef, useState } from 'react';
import type { SkPicture } from '@shopify/react-native-skia';
import {
  useDerivedValue,
  useFrameCallback,
  useReducedMotion,
  useSharedValue,
  withTiming,
  type DerivedValue,
  type SharedValue,
} from 'react-native-reanimated';
import { buildColorLUT } from './colors';
import { blendDots } from './engine/blend';
import { recordPicture } from './engine/paint';
import { MODES, precomputeCached } from './engine/registry';
import { quatToMat3 } from './engine/core';
import {
  acquireDotBuffer,
  acquireDotBufferB,
  acquireDotBufferC,
  acquireDynamics,
  acquireMat3,
  acquireOrient,
} from './engine/scratch';
import { correspondenceFor } from './engine/correspondence';
import { applyVoicePass } from './engine/voice-pass';
import type { VoiceBehaviour } from './engine/voice';
import { pickDesignSize, resolvePreset, resolveVoicePreset } from './presets';
import { useResolvedDark } from './theme';
import type { OrbBands, OrbState, ThinkingOrbProps } from './types';

// Cap the per-frame delta so a pause/resume or a dropped-frame hitch
// advances the phase by at most a few frames instead of the whole gap —
// the animation continues from its current pose without a visible jump.
const MAX_DT_MS = 100;
/**
 * Speed multiplier under reduced motion. The orb keeps animating rather
 * than freezing on a representative frame, because freezing destroys the
 * one thing it exists to communicate: `idle`, `listening` and `thinking`
 * share a 0.9 resting radius by design, so what separates them IS the
 * motion, and no still frame recovers it. The setting asks for less
 * motion, not none.
 *
 * A third of the pace keeps every state legible while removing the quick
 * gestures — the spike lunges, the fast `connecting` churn — that make
 * motion uncomfortable in the first place.
 */
const REDUCED_SPEED = 0.3;
/**
 * The amplitude a reduced-motion orb is drawn at. Pinning it to 0 froze
 * `listening` and `speaking` at their emptiest pose — a 0.9 shell with a
 * barely-there ripple — which is where they are least distinguishable
 * from `idle` and `thinking`, in a component whose entire job is state
 * indication. A representative level instead lets each state show the
 * pose it is recognised by: `speaking` sits visibly smaller (0.79 against
 * 0.9) with outward wavefronts, `listening` fuller with inward ones.
 *
 * The LIVE level is never read here. Voice-locked amplitude is fast,
 * irregular motion driven by something the viewer cannot anticipate,
 * which is precisely what the setting asks to be spared — so the level is
 * held constant and only the wavefronts drift, at {@linkcode
 * REDUCED_SPEED}.
 *
 * 0.85 rather than a peak: the benefit rises monotonically with level but
 * flattens, and 0.85 still reads as a real speech level.
 */
const REDUCED_AMP = 0.85;
/**
 * The band level a reduced-motion orb is drawn at, for the same reason and
 * by the same rule as {@linkcode REDUCED_AMP}: hold the level constant so
 * the shell keeps a shape, and stop it tracking the voice, which is the
 * fast irregular part the setting actually asks to be spared. Zeroing the
 * bands instead would remove the swell and ripple entirely — the same
 * "freeze the orb and destroy what distinguishes the states" mistake
 * REDUCED_AMP exists to correct, one layer up.
 *
 * Much lower than REDUCED_AMP because these terms compound: a constant
 * 0.85 across all three would sit the shell at a permanent full swell with
 * a standing ripple, which reads as a bug rather than as a resting state.
 * 0.3 is visible without being loud.
 */
const REDUCED_BANDS = 0.3;

// One-pole smoothing time constants. Speech onsets need to land almost
// immediately or the orb lags the voice; the decay is slower so syllable
// gaps read as a held breath rather than a stutter.
const ATTACK_MS = 45;
const RELEASE_MS = 240;

/**
 * One frame of the band filter: the same fast-attack / slow-release
 * one-pole the overall level uses, with the same "losing the source is a
 * target of 0, not an assignment of 0" rule so a band releases instead of
 * snapping when its source goes away.
 *
 * Returns `cur` untouched when there is no source and the band has already
 * settled — the cheap path every non-audio caller takes every frame.
 */
function filterBand(
  cur: number,
  sv: SharedValue<number> | undefined,
  dt: number
): number {
  'worklet';
  if (sv == null && cur === 0) return 0;
  let a = 0;
  if (sv != null) {
    // The SharedValue belongs to the caller; a NaN here would reach every
    // dot coordinate through the voice pass.
    a = sv.get();
    if (!(a > 0)) a = 0;
    else if (a > 1) a = 1;
  }
  const tau = a > cur ? ATTACK_MS : RELEASE_MS;
  const next = cur + (a - cur) * (1 - Math.exp(-dt / tau));
  return a === 0 && next < 1e-4 ? 0 : next;
}

/**
 * Options for {@linkcode useThinkingOrbPicture} — the animation subset of
 * {@linkcode ThinkingOrbProps} (everything except the container-only
 * `style` and `accessibilityLabel`).
 */
export type UseThinkingOrbPictureOptions = Omit<
  ThinkingOrbProps,
  'style' | 'accessibilityLabel'
> & {
  /**
   * Render the voice shell at this behaviour instead of `state`, blending
   * smoothly whenever it changes. Set by {@linkcode VoiceOrb}; `state` is
   * ignored while it is present.
   */
  voice?: VoiceBehaviour;

  /**
   * Live audio level, `0`–`1`, driving the voice shell's wavefronts. Only
   * meaningful alongside `voice` — the six ported animations have no audio
   * response, by design.
   *
   * Pass a `SharedValue` to write it at frame rate without re-rendering
   * React. Values are clamped and smoothed here (fast attack, slow
   * release), so feed a raw meter without pre-smoothing.
   */
  amplitude?: SharedValue<number> | number;

  /**
   * Three smoothed audio bands driving the voice pass — swell from `low`,
   * a travelling ripple from `mid`, ink from `high`. See
   * {@linkcode useVoiceLevels}, which produces them from raw PCM.
   *
   * Unlike {@linkcode amplitude} this is NOT voice-only: the pass runs
   * over the finished dot cloud, so it composes with all six ported
   * animations as well as the voice shell. Omit it, or leave the bands at
   * zero, and every mode paints exactly the pose it built.
   */
  bands?: OrbBands;
};

/**
 * How long a voice behaviour takes to blend into the next one. Tuned for
 * full-duplex sessions, where a barge-in flips speaking → listening and the
 * orb has to acknowledge the interruption immediately; long enough to read
 * as travel rather than a cut, short enough not to lag the conversation.
 */
const BLEND_MS = 280;

/**
 * How long a STATE change takes to travel, in ms — the six ported states,
 * not the voice behaviours above.
 *
 * Slower than {@linkcode BLEND_MS} on purpose. A voice blend is one shell
 * relaxing into a new behaviour during a conversation, where lagging the
 * turn is the worse failure. This one carries every dot from one geometry
 * to a different one — orbits to a rubik lattice — along paths that are
 * long because the two emission orders have nothing to do with each other.
 * Rushed, that reads as a scramble; given time, it reads as the cloud
 * rearranging itself, which is the point.
 */
const STATE_BLEND_MS = 560;

/** One turn in radians, for wrapping the orientation lock's yaw delta. */
const TAU = Math.PI * 2;

/** The two states a blend is travelling between; equal when settled. */
interface StatePair {
  from: OrbState;
  to: OrbState;
}

/**
 * Drive one orb's animation and return its per-frame Skia picture. The
 * picture is recorded with bounds `(0, 0, size, size)`; draw it in a
 * `<Picture>`, offset with a `<Group transform={...}>` when composing
 * several into one canvas.
 *
 * @see {@linkcode ThinkingOrb} for the drop-in component form.
 */
export function useThinkingOrbPicture({
  state = 'working',
  size = 64,
  theme = 'auto',
  speed = 1,
  paused = false,
  color,
  colorTo,
  colorShift,
  colorSpread = 0.6,
  colorCycleMs = 9000,
  amplitude,
  bands,
  tilt,
  voice,
  dotScale,
  debugFrameMs,
}: UseThinkingOrbPictureOptions = {}): DerivedValue<SkPicture> {
  const designSize = pickDesignSize(size);
  const isVoice = voice != null;

  // The two states the cloud is travelling between. Equal outside a blend,
  // and `resolvePreset` is cached, so `from === to` then yields the very
  // same object — which is exactly the test the single-build fast path in
  // the render loop uses.
  const [pair, setPair] = useState<StatePair>({ from: state, to: state });
  // Mirrors `pair` for the state-change effect, which needs the CURRENT
  // pair without listing it as a dependency: `from` is chosen from the
  // blend position at the moment of the change, so re-running the effect
  // when the pair it just set lands would restart the blend it started.
  const pairRef = useRef(pair);
  /** Blend position, 0 = fully `pair.from`, 1 = fully `pair.to`. */
  const stateMix = useSharedValue(1);

  const resolvedTo = useMemo(
    () =>
      isVoice
        ? resolveVoicePreset(designSize)
        : resolvePreset(pair.to, designSize),
    [isVoice, pair.to, designSize]
  );
  const resolvedFrom = useMemo(
    () =>
      isVoice
        ? resolveVoicePreset(designSize)
        : resolvePreset(pair.from, designSize),
    [isVoice, pair.from, designSize]
  );
  // Reference inequality IS the blend flag, by the caching contract above.
  // The voice shell never takes this path: its states are behaviours on one
  // profile that `buildVoice` already blends per dot, so dual-evaluating it
  // would pay for a second build to interpolate a pose against itself.
  const blending = resolvedFrom !== resolvedTo;

  const optsTo = resolvedTo.opts;
  const optsFrom = resolvedFrom.opts;
  const rMinTo = optsTo.rMin ?? 0.3;
  const rMinFrom = optsFrom.rMin ?? 0.3;

  const buildTo = MODES[resolvedTo.mode].build;
  const buildFrom = MODES[resolvedFrom.mode].build;
  // Absent for `morph`, which never projects — the lock then does nothing,
  // which is correct: a flat outline has no orientation to reconcile.
  const orientTo = MODES[resolvedTo.mode].orient;
  const orientFrom = MODES[resolvedFrom.mode].orient;
  const staticTo = useMemo(
    () => precomputeCached(resolvedTo.mode, optsTo),
    [resolvedTo.mode, optsTo]
  );
  // Only precomputed when the two halves genuinely differ. Two unconditional
  // memos would build the SAME tables twice on every mount and keep both
  // resident — including for every `VoiceOrb`, which resolves one cached
  // preset and so can never blend this way. Outside a blend this aliases
  // `staticTo`, which is exactly right: one dataset, one build.
  const staticFromOwn = useMemo(
    () =>
      resolvedFrom === resolvedTo
        ? null
        : precomputeCached(resolvedFrom.mode, optsFrom),
    [resolvedFrom, resolvedTo, optsFrom]
  );
  const staticFrom = staticFromOwn ?? staticTo;
  // Which dot becomes which. Resolved once per pair of clouds and cached, so
  // the position matching never runs on the state-change path — see
  // `engine/correspondence.ts`.
  const corr = useMemo(
    () => correspondenceFor(staticFrom, staticTo),
    [staticFrom, staticTo]
  );

  const dark = useResolvedDark(theme);
  const lut = useMemo(() => buildColorLUT(dark, color), [dark, color]);
  // Built only when a second endpoint exists; `undefined` is the signal
  // the painter uses to take the original single-ramp path.
  const lutTo = useMemo(
    () => (colorTo == null ? undefined : buildColorLUT(dark, colorTo)),
    [dark, colorTo]
  );

  // Same number-or-SharedValue handling as amplitude. `undefined` here
  // means "nobody is driving it", which is what selects the built-in drift.
  const ownShiftSV = useSharedValue(0);
  useEffect(() => {
    if (typeof colorShift === 'number') ownShiftSV.set(colorShift);
  }, [colorShift, ownShiftSV]);
  const shiftSV = typeof colorShift === 'number' ? ownShiftSV : colorShift;

  // Globe rotation, same number-or-SharedValue handling again.
  const ownYawSV = useSharedValue(0);
  const ownPitchSV = useSharedValue(0);
  const ownRollSV = useSharedValue(0);
  const tYaw = tilt?.yaw;
  const tPitch = tilt?.pitch;
  const tRoll = tilt?.roll;
  useEffect(() => {
    if (typeof tYaw === 'number') ownYawSV.set(tYaw);
    if (typeof tPitch === 'number') ownPitchSV.set(tPitch);
    if (typeof tRoll === 'number') ownRollSV.set(tRoll);
  }, [tYaw, tPitch, tRoll, ownYawSV, ownPitchSV, ownRollSV]);
  const yawSV = typeof tYaw === 'number' ? ownYawSV : tYaw;
  const pitchSV = typeof tPitch === 'number' ? ownPitchSV : tPitch;
  const rollSV = typeof tRoll === 'number' ? ownRollSV : tRoll;
  // The globe's own orientation, if the caller is driving one. Always four
  // shared values or nothing — unlike the angles above there is no plain
  // number form, because a static orientation is just a different resting
  // pose and the three angles already express that.
  const qx = tilt?.orientation?.x;
  const qy = tilt?.orientation?.y;
  const qz = tilt?.orientation?.z;
  const qw = tilt?.orientation?.w;
  const reduced = useReducedMotion();

  // One clock per half of the blend. A single speed would advance the
  // OUTGOING animation at the incoming state's tempo for the length of the
  // blend, which reads as the old animation changing pace as it leaves.
  const userSpeed = speed * (reduced ? REDUCED_SPEED : 1);
  const effSpeedTo = resolvedTo.speed * userSpeed;
  const effSpeedFrom = resolvedFrom.speed * userSpeed;
  const effSpeedToSV = useSharedValue(effSpeedTo);
  const effSpeedFromSV = useSharedValue(effSpeedFrom);
  useEffect(() => {
    effSpeedToSV.set(effSpeedTo);
    effSpeedFromSV.set(effSpeedFrom);
  }, [effSpeedTo, effSpeedFrom, effSpeedToSV, effSpeedFromSV]);

  // Amplitude arrives either as a caller-owned SharedValue (driven at
  // frame rate from the UI thread) or as a plain number mirrored into our
  // own — the worklet below only ever reads one SharedValue either way.
  const ownAmpSV = useSharedValue(0);
  useEffect(() => {
    if (typeof amplitude === 'number') ownAmpSV.set(amplitude);
  }, [amplitude, ownAmpSV]);
  const ampSV = typeof amplitude === 'number' ? ownAmpSV : amplitude;

  // Dot weight, same number-or-SharedValue handling. Seeded at 1 rather than 0:
  // this one is a MULTIPLIER, so the neutral value is one, and a caller who
  // never passes it must not have their dots collapse to the `rMin` floor.
  const ownDotScaleSV = useSharedValue(1);
  useEffect(() => {
    if (typeof dotScale === 'number') ownDotScaleSV.set(dotScale);
  }, [dotScale, ownDotScaleSV]);
  const dotScaleSV = typeof dotScale === 'number' ? ownDotScaleSV : dotScale;

  // Same number-or-SharedValue handling for each band. Three own-values are
  // allocated unconditionally: hooks cannot be called per present band, and
  // an unused SharedValue is a single UI-thread slot.
  const ownLowSV = useSharedValue(0);
  const ownMidSV = useSharedValue(0);
  const ownHighSV = useSharedValue(0);
  const bLow = bands?.low;
  const bMid = bands?.mid;
  const bHigh = bands?.high;
  useEffect(() => {
    if (typeof bLow === 'number') ownLowSV.set(bLow);
    if (typeof bMid === 'number') ownMidSV.set(bMid);
    if (typeof bHigh === 'number') ownHighSV.set(bHigh);
  }, [bLow, bMid, bHigh, ownLowSV, ownMidSV, ownHighSV]);
  const lowSV = typeof bLow === 'number' ? ownLowSV : bLow;
  const midSV = typeof bMid === 'number' ? ownMidSV : bMid;
  const highSV = typeof bHigh === 'number' ? ownHighSV : bHigh;

  // The smoothed level, 0–1, read per dot by the voice shell. Losing the
  // amplitude source is a release, not a reset — see the frame callback.
  const level = useSharedValue(0);

  // The three band levels, smoothed through the same filter as `level` but
  // consumed by the voice pass (`engine/voice-pass.ts`) rather than by the
  // shell. Unlike `amplitude`, these apply to EVERY mode: the pass runs
  // over the finished dot cloud, so a caller can make any of the six
  // ported animations audio-reactive without switching to the voice shell.
  const bandLow = useSharedValue(0);
  const bandMid = useSharedValue(0);
  const bandHigh = useSharedValue(0);

  // Voice behaviour blending. `from`/`to` are behaviour indices and `mix`
  // walks 0 → 1 across a state change; the mode evaluates both and
  // interpolates per dot. A change landing mid-blend restarts from the
  // previous TARGET rather than the current interpolated pose — states
  // change on human timescales, so this is rare and barely visible.
  const behFrom = useSharedValue(voice ?? 0);
  const behTo = useSharedValue(voice ?? 0);
  const mix = useSharedValue(1);
  useEffect(() => {
    if (voice == null) return;
    if (behTo.get() === voice) return;
    // A change arriving MID-blend cannot resume from the pose on screen:
    // that pose is a lerp of two behaviours, and `from` can only name one
    // of them. Start from whichever endpoint the pose is currently nearer
    // — that halves the worst-case jump, from |B−A| to 0.5·|B−A|, against
    // always taking the outgoing target. Removing the jump entirely needs
    // a three-way blend; see plans/002.
    if (mix.get() >= 0.5) behFrom.set(behTo.get());
    behTo.set(voice);
    mix.set(0);
    mix.set(withTiming(1, { duration: BLEND_MS }));
  }, [voice, behFrom, behTo, mix]);

  // One clock per half of the blend. -1 marks a clock as unseeded; the next
  // active frame seeds it from the shared frame clock so instances mounted
  // at different times lock.
  //
  // Neither is reset by a state change. `phaseTo` in particular runs
  // unbroken for the component's whole life, because it is also the clock
  // the colour drift and the voice pass read: reseeding it on every state
  // change would jump the hue and the ripple at the exact moment the dots
  // are supposed to be travelling smoothly. Only a design-size change,
  // which is a resize and a discontinuity already, reseeds them.
  const phaseTo = useSharedValue(-1);
  const phaseFrom = useSharedValue(-1);
  useEffect(() => {
    phaseTo.set(-1);
    phaseFrom.set(-1);
  }, [isVoice, designSize, phaseTo, phaseFrom]);

  // State blending. Unlike the voice blend above — one shell, two behaviour
  // indices, interpolated inside `buildVoice` — the six states are six
  // different mode implementations, so the blend happens on their OUTPUTS
  // (`engine/blend.ts`) and both halves need their own preset and clock.
  useEffect(() => {
    if (isVoice) return;
    const p = pairRef.current;
    if (p.to === state) return;
    // Same rule, and the same caveat, as the voice blend: a change arriving
    // mid-blend cannot resume from the pose on screen, because that pose is
    // a lerp of two states and `from` can only name one of them. Starting
    // from whichever endpoint it is nearer halves the worst-case jump.
    const takeTo = stateMix.get() >= 0.5;
    // The outgoing half inherits the clock the incoming one has been
    // running, so the state being left behind keeps animating from the pose
    // it was actually in rather than restarting.
    if (takeTo) phaseFrom.set(phaseTo.get());
    const next: StatePair = { from: takeTo ? p.to : p.from, to: state };
    pairRef.current = next;
    setPair(next);
  }, [isVoice, state, stateMix, phaseFrom, phaseTo]);

  // The travel is started SEPARATELY, keyed on the pair, so it begins on the
  // render that installs the new builders rather than the one that asks for
  // them. Starting it alongside `setPair` above ran the clock through
  // React's render and commit: the first thing drawn was already part-way
  // through the blend, and with a long commit most of the travel had
  // happened before anything could show it — the change read as a stutter
  // followed by a late, truncated move.
  useEffect(() => {
    if (pair.from === pair.to) {
      // Nothing to travel — a toggle that landed back on the state it came
      // from. Park the mix at the end so the "nearer endpoint" test above
      // reads a settled blend rather than a stale mid-flight value.
      stateMix.set(1);
      return;
    }
    stateMix.set(0);
    stateMix.set(withTiming(1, { duration: STATE_BLEND_MS }));
  }, [pair, stateMix]);

  const frame = useFrameCallback((info) => {
    'worklet';
    let dt = info.timeSincePreviousFrame ?? 0;
    if (dt > MAX_DT_MS) dt = MAX_DT_MS;
    const now = info.timestamp / 1000;
    const step = dt / 1000;
    // Seeding no longer returns early: `phaseFrom` can be seeded mid-life
    // by a state change, and skipping the amplitude filter for that frame
    // would stall the voice level on exactly the transitions it smooths.
    // The first frame carries dt 0, so the filter below is a no-op there.
    if (phaseTo.get() < 0) phaseTo.set(now * effSpeedToSV.get());
    else phaseTo.set(phaseTo.get() + step * effSpeedToSV.get());
    if (phaseFrom.get() < 0) phaseFrom.set(now * effSpeedFromSV.get());
    else phaseFrom.set(phaseFrom.get() + step * effSpeedFromSV.get());

    const cur = level.get();
    // Losing the source is a target of 0, not an assignment of 0. Every
    // state but `listening`/`speaking` supplies no amplitude, so
    // `speaking → thinking` — the most common transition in the machine —
    // drops it; snapping the level there would bypass the RELEASE_MS
    // release on precisely the transition it exists for. The `cur !== 0`
    // arm keeps the six non-voice animations out of the filter entirely.
    if (ampSV != null || cur !== 0) {
      let a = 0;
      if (ampSV != null) {
        // Clamp defensively: the SharedValue belongs to the caller, and a
        // NaN would propagate into every dot coordinate.
        a = ampSV.get();
        if (!(a > 0)) a = 0;
        else if (a > 1) a = 1;
      }
      // One-pole filter, frame-rate independent via the exponential — a
      // dropped frame lands in the same place as several short ones.
      const tau = a > cur ? ATTACK_MS : RELEASE_MS;
      const next = cur + (a - cur) * (1 - Math.exp(-dt / tau));
      // An exponential never quite reaches its target, so settle the tail
      // exactly to 0 — otherwise the branch above can never switch off.
      level.set(a === 0 && next < 1e-4 ? 0 : next);
    }

    // The three bands run the same filter, each with the same guard: a band
    // with no source and a settled 0 costs nothing, which is what keeps the
    // six ported modes untouched for callers who drive no audio.
    bandLow.set(filterBand(bandLow.get(), lowSV, dt));
    bandMid.set(filterBand(bandMid.get(), midSV, dt));
    bandHigh.set(filterBand(bandHigh.get(), highSV, dt));
  }, false);

  useEffect(() => {
    // Reduced motion slows the clock (see REDUCED_SPEED); it does not stop
    // it. Only `paused` and the frozen `failed` state stop the clock.
    frame.setActive(!paused);
  }, [paused, frame]);

  // Sized for the LARGER of the two clouds: the blend carries
  // `max(nFrom, nTo)` dots so neither pose arrives thinned out, and both
  // buffers are acquired at that capacity so the in-place lerp has room.
  const dotCount =
    staticFrom.dotCount > staticTo.dotCount
      ? staticFrom.dotCount
      : staticTo.dotCount;

  return useDerivedValue(() => {
    const t = Math.max(0, phaseTo.get());
    // Dot weight for THIS frame. Floored just above zero rather than clamped to
    // a design range: a caller animating it is free to choose the range, but a 0
    // would make every dot vanish into the `rMin` floor and read as the orb
    // having failed to draw.
    const rMulRaw = dotScaleSV == null ? 1 : dotScaleSV.get();
    const rMul = rMulRaw > 0.01 ? rMulRaw : 0.01;
    // High-res timer polyfilled on the UI runtime; read via globalThis so
    // no ambient `performance` global leaks into the published types.
    const perf = (globalThis as { performance?: { now(): number } })
      .performance;
    const timed = debugFrameMs != null && perf != null;
    const t0 = timed && perf ? perf.now() : 0;
    // Reduce-motion holds the level constant — the shell keeps its shape
    // but stops tracking the voice, which is the fast irregular part. See
    // REDUCED_AMP.
    const amp = reduced ? REDUCED_AMP : level.get();
    const buf = acquireDotBuffer(dotCount);
    // Where the finished cloud ends up: the single build writes straight
    // into `buf`, a blend produces its own buffer. Everything downstream
    // reads this one.
    let pose = buf;
    // Blends run under reduced motion too. `mix` is driven by withTiming,
    // independently of the frame callback, and cutting between behaviours
    // instead would be a harder visual event than the travel it replaces.
    // Reduced motion keeps the globe level: this rotation is driven by the
    // device, so it is the fast, irregular, unanticipatable kind of motion
    // the setting exists to remove, and it carries no state.
    const dyn = acquireDynamics(
      amp,
      behFrom.get(),
      behTo.get(),
      mix.get(),
      reduced || yawSV == null ? 0 : yawSV.get(),
      reduced || pitchSV == null ? 0 : pitchSV.get(),
      reduced || rollSV == null ? 0 : rollSV.get(),
      // Orientation SURVIVES reduced motion, where the three angles above do
      // not, and the difference is who caused it. Those are ambient: the
      // device moves and the globe answers, unbidden, which is the motion the
      // setting exists to remove. An orientation is where the user themselves
      // put the ball. Zeroing it would not calm the interface, it would
      // silently discard a direct manipulation — and the caller is expected to
      // drop the coasting half itself, which is the part that keeps moving
      // after the finger has gone.
      qx == null || qy == null || qz == null || qw == null
        ? null
        : quatToMat3(qx.get(), qy.get(), qz.get(), qw.get(), acquireMat3()),
      rMul
    );
    // Blend position for this frame. Pinned to 1 whenever the two halves
    // resolve to the same preset, which is every frame outside a state
    // change — so a settled orb runs exactly one build, as it always did.
    let m = 1;
    if (blending) {
      const sm = stateMix.get();
      m = !(sm > 0) ? 0 : sm > 1 ? 1 : sm;
    }
    if (m >= 1) {
      buildTo(buf, size, t, optsTo, staticTo, dyn);
    } else {
      const tFrom = Math.max(0, phaseFrom.get());
      // ORIENTATION LOCK. The blend interpolates projected screen points,
      // which is only meaningful if both clouds were drawn looking at the
      // orb from the same angle. Otherwise a dot sits on opposite sides in
      // the two poses, the straight line between them passes through the
      // middle, and the shell collapses into tiers — `globe` spins at
      // `t*0.5` against `wave`'s `t*0.18`, so that pair collapsed hardest.
      //
      // Both halves are therefore pulled onto ONE orientation for the
      // frame: the interpolated one. Each mode is nudged by the difference
      // between that common angle and its own, which is zero at its own end
      // of the blend — so neither cloud jumps as it takes over, and in
      // between the two agree.
      let lockYaw = 0;
      let lockPitch = 0;
      if (orientFrom != null && orientTo != null) {
        const oa = acquireOrient(0);
        const ob = acquireOrient(1);
        orientFrom(tFrom, optsFrom, oa);
        orientTo(t, optsTo, ob);
        // Yaw grows without bound with elapsed time and the two modes run
        // at very different rates, so the raw difference is routinely dozens
        // of turns. Wrap it to the shorter way round, or the orb spins
        // itself into a blur over the length of the blend.
        let dYaw = ob[0]! - oa[0]!;
        dYaw -= Math.round(dYaw / TAU) * TAU;
        lockYaw = dYaw;
        lockPitch = ob[1]! - oa[1]!;
      }
      // `from` turns TOWARD the common angle as the blend runs; `to` starts
      // turned back from it by the same amount and unwinds. Added to
      // whatever the caller is driving rather than replacing it, so a device
      // tilt still applies throughout.
      const baseYaw = dyn.yaw;
      const basePitch = dyn.pitch;
      dyn.yaw = baseYaw + lockYaw * m;
      dyn.pitch = basePitch + lockPitch * m;
      buildFrom(buf, size, tFrom, optsFrom, staticFrom, dyn);

      dyn.yaw = baseYaw - lockYaw * (1 - m);
      dyn.pitch = basePitch - lockPitch * (1 - m);
      // Its own `globalThis` slot, so this build cannot overwrite the pose
      // the one above just laid down.
      const bufB = acquireDotBufferB(dotCount);
      buildTo(bufB, size, t, optsTo, staticTo, dyn);

      dyn.yaw = baseYaw;
      dyn.pitch = basePitch;
      // A third buffer: a position-matched pairing can read a source index
      // above the one being written, so the result cannot go back over
      // either input.
      pose = acquireDotBufferC(corr.n);
      blendDots(pose, buf, bufB, m, corr);
    }
    // The dot-radius floor rides the blend too — it is per-preset (the
    // morph outline sits lower than the rest), so holding it at one
    // endpoint would clamp the other's dots for the length of the travel.
    const rMin = rMinFrom + (rMinTo - rMinFrom) * m;
    // The voice pass runs on the BUILT cloud, which is what lets it apply
    // to every mode rather than to the voice shell alone.
    //
    // Reduced motion holds the bands constant rather than zeroing them —
    // the same rule as REDUCED_AMP, and for the same reason. Zeroing would
    // drop the swell and ripple entirely, which is the "freeze it and lose
    // what tells the states apart" failure that decision already corrects
    // for the shell. Held constant, the shape survives and only the
    // wavefront drifts, at REDUCED_SPEED. Bands still present or settled at
    // zero decide WHETHER audio is driving the orb at all, so a silent
    // caller stays a no-op here too.
    const driven = bandLow.get() > 0 || bandMid.get() > 0 || bandHigh.get() > 0;
    if (reduced) {
      if (driven) {
        applyVoicePass(
          pose,
          size,
          t,
          REDUCED_BANDS,
          REDUCED_BANDS,
          REDUCED_BANDS
        );
      }
    } else {
      applyVoicePass(
        pose,
        size,
        t,
        bandLow.get(),
        bandMid.get(),
        bandHigh.get()
      );
    }
    // Where the cloud sits between the two ramps. A caller-driven shift
    // wins; otherwise the orb drifts on its own clock. Under reduced
    // motion the drift is pinned to the middle of the gradient — a colour
    // cycle is motion, and this is the one place it can be removed without
    // costing a state signal, since the hue carries none.
    let shift = 0;
    if (lutTo !== undefined) {
      if (shiftSV != null) {
        const s = shiftSV.get();
        shift = !(s > 0) ? 0 : s > 1 ? 1 : s;
      } else if (reduced) {
        shift = 0.5;
      } else {
        shift = 0.5 + 0.5 * Math.sin((t * 2 * Math.PI * 1000) / colorCycleMs);
      }
    }
    // `rMin` rides `rMul` too. It is the floor the painter clamps every radius
    // to, resolved on the JS thread from the profile — leave it fixed and a
    // shrinking multiplier stops thinning the dots the moment they reach it,
    // which looks like the animation sticking partway.
    const pic = recordPicture(
      pose,
      size,
      lut,
      rMin * rMul,
      lutTo,
      shift,
      colorSpread
    );
    if (timed && perf && debugFrameMs != null) {
      debugFrameMs.set(perf.now() - t0);
    }
    return pic;
  }, [
    // Both halves of the blend, in full. `ModeBuild` types its
    // `staticData` as `any`, so a builder left paired with the other
    // half's precomputed tables would not be caught by the compiler — it
    // would read whatever fields happened to line up and draw nonsense.
    buildTo,
    buildFrom,
    optsTo,
    optsFrom,
    staticTo,
    staticFrom,
    blending,
    corr,
    orientTo,
    orientFrom,
    dotCount,
    lut,
    lutTo,
    colorSpread,
    colorCycleMs,
    size,
    rMinTo,
    rMinFrom,
    reduced,
    debugFrameMs,
  ]);
}
