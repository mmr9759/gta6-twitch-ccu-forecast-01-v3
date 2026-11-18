// GTA VI launch-day Twitch CCU forecaster, v3.
//
// Idea (from the kit's own practice data, 38 past big launches + 4 practice
// launches): the launch-day curve of a game on Twitch is almost entirely
// pinned down by ONE number, its hour-1 print. Across every past launch,
// log(peak / hour-1) falls almost linearly in log(hour-1) (correlation about
// -0.94): a launch that is already huge in hour 1 is already near its peak,
// while a launch that is still small in hour 1 ramps several-fold over the
// day as time zones unlock (US East joins at hour 6, US West at hour 9) and
// the evening audience arrives. So this forecaster regresses, hour by hour,
// log(viewers_h / viewers_1) on log(viewers_1) over the comparable launches
// shipped in the case file, and blends three views of that regression:
//
//   1. all-pool size regression: every usable comparable launch;
//   2. rolling-midnight size regression: only launches whose peak is at most
//      4x their hour-1 print -- the regime GTA VI itself is in (it unlocks
//      at local midnight, so the UK, Europe, New Zealand and Australia are
//      already watching in hour 1, and the Americas join later);
//   3. kernel local-linear: the same regression weighted by closeness in
//      log size, so a huge launch is predicted mostly from huge launches.
//
// The three forecasts are blended in log space (0.35 / 0.35 / 0.30). Each
// component's quantiles come from its own regression residuals (normal
// quantiles, residual spread shrunk by 0.60 -- blending already removes
// part of the median error, so the honest band is narrower than any single
// component's). Slopes are shrunk toward zero and capped, so an hour-1
// print far outside the comparables' range (GTA VI's will be) extrapolates
// gently instead of exploding.
//
// What this deliberately does NOT do: it never hard-codes a practice
// launch's answer. The practice cases' curves are only used the way the
// final case uses them -- as comparable launches present in the case file
// at run time. The program reads nothing but its case file, uses no
// network, and is fully deterministic.
//
// Backtests (pinball loss, log scale; lower is better). Scorer validated
// against the kit: the house persistence baseline reproduces the RULES
// figures exactly (loss 0.387, typical miss 82.8%, 90% coverage 0.783).
//   - leave-one-out over the 38 comparables: 0.56 overall, 0.23 on
//     rolling-regime launches (the GTA VI regime; the overall figure is
//     dominated by simultaneous-global launches whose hour 1 is near
//     zero). House baseline on the same split: 0.93 overall.
//   - prac-like (run on p00-p03, scored against SullyGnome's hourly
//     averages, which match the cases' hour-1 prints exactly): 0.190.
//     v2 scored 0.200 on the board; v1 0.234.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];

/** Slope shrinkage: a fit on n launches keeps n / (n + SHRINK) of its slope. */
const SHRINK = 8;
/** Cap on |slope| of log-ratio against log size, per hour. */
const SLOPE_CAP = 0.35;
/** Residual spread is floored / capped, then multiplied by SD_MULT. */
const SD_FLOOR = 0.12;
const SD_CAP = 1.2;
const SD_MULT = 0.6;
/** Gaussian kernel bandwidth (log-size units) for the local-linear component. */
const KERNEL_BW = 1.5;
/** Blend weights: all-pool regression, rolling-only regression, kernel. */
const W_ALL = 0.35;
const W_ROLL = 0.35;
const W_KER = 0.3;
/** A comparable counts as "rolling-midnight" when its peak is at most this multiple of hour 1. */
const ROLL_PEAK_MAX = 4;

interface HourlyPoint {
  from: string;
  viewers: number;
}

interface Comparable {
  name: string;
  day_start: string;
  hourly: HourlyPoint[];
}

interface CaseFile {
  levels: number[];
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  comparables: Comparable[];
}

/** One comparable launch reduced to its release-day curve in log-ratio space. */
interface LaunchCurve {
  logV1: number;
  peakRatio: number;
  /** log(viewers_h / viewers_1) by hour, hours 1..24 where present. */
  lr: Map<number, number>;
}

/** x rounded to 2 decimals (half up). */
const round2 = (x: number): number => Math.round(x * 100) / 100;

/** Population standard deviation. */
const pstdev = (xs: number[]): number => {
  if (xs.length < 2) return 0.3;
  const m = xs.reduce((a, b) => a + b, 0) / xs.length;
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) * (b - m), 0) / xs.length);
};

/** Ordinary least squares of ys on xs: { meanY, slope, meanX }. */
const linfit = (xs: number[], ys: number[]): { meanY: number; slope: number; meanX: number } => {
  const n = xs.length;
  const meanX = xs.reduce((a, b) => a + b, 0) / n;
  const meanY = ys.reduce((a, b) => a + b, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - meanX) * (xs[i] - meanX);
    sxy += (xs[i] - meanX) * (ys[i] - meanY);
  }
  return { meanY, slope: sxx > 0 ? sxy / sxx : 0, meanX };
};

/** Weighted least squares (same return shape as linfit). */
const wlinfit = (xs: number[], ys: number[], ws: number[]): { meanY: number; slope: number; meanX: number } => {
  const tw = ws.reduce((a, b) => a + b, 0);
  const meanX = xs.reduce((a, x, i) => a + ws[i] * x, 0) / tw;
  const meanY = ys.reduce((a, y, i) => a + ws[i] * y, 0) / tw;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < xs.length; i++) {
    sxx += ws[i] * (xs[i] - meanX) * (xs[i] - meanX);
    sxy += ws[i] * (xs[i] - meanX) * (ys[i] - meanY);
  }
  return { meanY, slope: sxx > 0 ? sxy / sxx : 0, meanX };
};

/** 'YYYY-MM-DDTHH:00:00Z' for dayStart + (hour - 1) hours. */
const hourKey = (dayStartMs: number, hour: number): string => {
  const d = new Date(dayStartMs + (hour - 1) * 3600_000);
  return d.toISOString().slice(0, 19) + 'Z';
};

/**
 * Reduce the case's comparables to usable launch curves. A launch is usable
 * when its release day has an hour-1 print and at least 20 of its 24 hourly
 * prints. The Grand Theft Auto VI category's own pre-release history is not
 * a launch curve (its game has not launched yet), so it is excluded by name.
 */
const buildCurves = (c: CaseFile): LaunchCurve[] => {
  const curves: LaunchCurve[] = [];
  for (const comp of c.comparables ?? []) {
    if (comp.name.includes('Grand Theft Auto VI')) continue;
    const byHour = new Map<string, number>();
    for (const p of comp.hourly ?? []) {
      if (typeof p.viewers === 'number' && p.viewers > 0) byHour.set(p.from, p.viewers);
    }
    const dayStartMs = Date.parse(comp.day_start);
    if (Number.isNaN(dayStartMs)) continue;
    const vals = new Map<number, number>();
    for (let h = 1; h <= 24; h++) {
      const v = byHour.get(hourKey(dayStartMs, h));
      if (v !== undefined) vals.set(h, v);
    }
    const v1 = vals.get(1);
    if (v1 === undefined || vals.size < 20) continue;
    let peak = 0;
    for (const v of vals.values()) if (v > peak) peak = v;
    const lr = new Map<number, number>();
    for (const [h, v] of vals) lr.set(h, Math.log(v / v1));
    curves.push({ logV1: Math.log(v1), peakRatio: peak / v1, lr });
  }
  return curves;
};

/** Quantiles of one hour from a regression's fitted centre mu and spread sd. */
const quantilesFrom = (mu: number, sd: number, v1: number): number[] =>
  Z.map((z) => Math.exp(mu + z * sd) * v1);

/**
 * Size-regression forecast for one hour: OLS of log-ratio on log size over
 * the given launches, slope shrunk and capped, spread from the residuals.
 * Returns null when too few launches have that hour.
 */
const regressHour = (pool: LaunchCurve[], hour: number, logV1: number, v1: number): number[] | null => {
  const xs: number[] = [];
  const ys: number[] = [];
  for (const p of pool) {
    const y = p.lr.get(hour);
    if (y !== undefined) {
      xs.push(p.logV1);
      ys.push(y);
    }
  }
  if (xs.length < 5) return null;
  const { meanY, slope: rawSlope, meanX } = linfit(xs, ys);
  const slope = Math.max(-SLOPE_CAP, Math.min(SLOPE_CAP, rawSlope * (xs.length / (xs.length + SHRINK))));
  const mu = meanY + slope * (logV1 - meanX);
  const resid = ys.map((y, i) => y - (meanY + slope * (xs[i] - meanX)));
  const sd = Math.min(SD_CAP, Math.max(SD_FLOOR, pstdev(resid))) * SD_MULT;
  return quantilesFrom(mu, sd, v1);
};

/**
 * Kernel local-linear forecast for one hour: the same regression, but each
 * launch is weighted by a Gaussian kernel in log size around the target, so
 * a huge launch is predicted mostly from huge launches. Returns null when
 * the effective sample is too small.
 */
const kernelHour = (pool: LaunchCurve[], hour: number, logV1: number, v1: number): number[] | null => {
  const xs: number[] = [];
  const ys: number[] = [];
  const ws: number[] = [];
  for (const p of pool) {
    const y = p.lr.get(hour);
    if (y === undefined) continue;
    xs.push(p.logV1);
    ys.push(y);
    ws.push(Math.exp(-0.5 * ((p.logV1 - logV1) / KERNEL_BW) ** 2));
  }
  const tw = ws.reduce((a, b) => a + b, 0);
  const neff = tw * tw / ws.reduce((a, w) => a + w * w, 0);
  if (xs.length < 5 || neff < 3) return null;
  const { meanY, slope: rawSlope, meanX } = wlinfit(xs, ys, ws);
  const slope = Math.max(-SLOPE_CAP, Math.min(SLOPE_CAP, rawSlope * (neff / (neff + SHRINK))));
  const mu = meanY + slope * (logV1 - meanX);
  const resid = ys.map((y, i) => y - (meanY + slope * (xs[i] - meanX)));
  const sd = Math.min(SD_CAP, Math.max(SD_FLOOR, pstdev(resid))) * SD_MULT;
  return quantilesFrom(mu, sd, v1);
};

/** House persistence baseline (the kit starter): every hour stays at hour 1, spread widening like a random walk. */
const houseHour = (v1: number, hour: number): number[] =>
  Z.map((z) => round2(v1 * Math.exp(z * 0.15 * Math.sqrt(hour - 1))));

/** Blend one hour's component forecasts in log space; missing components are dropped and weights renormalised. */
const blendHour = (parts: Array<{ q: number[] | null; w: number }>, v1: number, hour: number): number[] => {
  const use = parts.filter((p) => p.q !== null) as Array<{ q: number[]; w: number }>;
  if (use.length === 0) return houseHour(v1, hour);
  const tw = use.reduce((a, p) => a + p.w, 0);
  const out: number[] = [];
  for (let j = 0; j < 5; j++) {
    let s = 0;
    for (const p of use) s += (p.w / tw) * Math.log(Math.max(p.q[j], 1e-9));
    out.push(Math.exp(s));
  }
  return out;
};

/** Forecast all 23 hours for one case. */
const forecastCase = (c: CaseFile): Array<{ hour: number; q: number[] }> => {
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error('case has no hour-1 print');
  const v1 = Math.max(1, first.viewers);
  const logV1 = Math.log(v1);
  const pool = buildCurves(c);
  const rolling = pool.filter((p) => p.peakRatio <= ROLL_PEAK_MAX);
  const hours = [...c.forecast].map((f) => f.hour).sort((a, b) => a - b);
  return hours.map((hour) => {
    let q: number[];
    if (pool.length < 5) {
      q = houseHour(v1, hour);
    } else {
      q = blendHour(
        [
          { q: regressHour(pool, hour, logV1, v1), w: W_ALL },
          { q: regressHour(rolling.length >= 5 ? rolling : pool, hour, logV1, v1), w: W_ROLL },
          { q: kernelHour(pool, hour, logV1, v1), w: W_KER },
        ],
        v1,
        hour,
      );
    }
    // Clean the row: finite, in range, non-decreasing, 2 decimals.
    const clean = q.map((x) => {
      if (!Number.isFinite(x)) return v1;
      return Math.min(1e9, Math.max(0.01, round2(x)));
    });
    for (let j = 1; j < 5; j++) if (clean[j] < clean[j - 1]) clean[j] = clean[j - 1];
    return { hour, q: clean };
  });
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const hours = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
