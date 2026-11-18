# GTA VI Twitch CCU forecaster — v3

Forecasts hour-by-hour Twitch concurrent viewers for Grand Theft Auto VI on
launch day (19 Nov 2026 UTC), as five quantiles per hour (hours 2–24), from
the hour-1 print and the comparable past launches shipped in the case file.

## Model

The launch-day curve is almost entirely pinned down by the hour-1 print:
across every past launch in the kit, log(peak / hour-1) falls almost
linearly in log(hour-1) (correlation ≈ −0.94). A launch already huge in
hour 1 is already near its peak; a launch still small in hour 1 ramps
several-fold as time zones unlock (US East at hour 6, US West at hour 9)
and the evening audience arrives.

`main.ts` (Node 22, stdlib only, no network, deterministic) regresses, hour
by hour, log(viewers_h / viewers_1) on log(viewers_1) over the comparables,
and blends three views in log space (0.35 / 0.35 / 0.30):

1. **All-pool size regression** — every usable comparable launch.
2. **Rolling-midnight size regression** — only launches whose peak is at
   most 4× their hour-1 print, the regime GTA VI is in (local-midnight
   unlock: UK, Europe, NZ/AU already watching in hour 1).
3. **Kernel local-linear** — the same regression weighted by closeness in
   log size (bandwidth 1.5), so a huge launch is predicted mostly from
   huge launches.

Quantiles come from each component's regression residuals (normal
quantiles, spread × 0.60 — blending already removes part of the median
error). Slopes are shrunk toward zero (n/(n+8)) and capped at ±0.35, so an
hour-1 print far outside the comparables' range extrapolates gently.
The GTA VI category's own pre-release history is excluded from the pool
(it is not a launch curve). If fewer than 5 comparables are usable, the
entry falls back to the house persistence baseline.

## Backtests (pinball loss, log scale)

Scorer validated against the kit: the house baseline reproduces the RULES
figures exactly (loss 0.387, typical miss 82.8%, 90% coverage 0.783).

- **Prac-like** (run on p00–p03, scored against SullyGnome hourly
  averages): **0.190** (v2: 0.200 on the board; v1: 0.234).
- **Leave-one-out** over the 38 comparables: 0.56 overall, 0.23 on
  rolling-regime launches (the GTA VI regime); house baseline 0.93.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads every `*.in` in the input dir, writes one `*.out` per case to the
output dir. Two runs are byte-identical.
