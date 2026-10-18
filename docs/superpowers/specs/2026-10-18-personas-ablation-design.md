# Do the working styles make the pool robust? Personas ablation: design

**Date:** 2026-10-18 · **Code:** `examples/contagion_net/` · **Batch:** `BATCHES=personas` in `run_long_pilot.sh` · **Follows:** the weekend run ([memory-file spec](2026-10-15-memory-file-design.md), addendum 8)

## Question

In the weekend run, the procedures arm at k = 24 gave Δ honest believers vs twin of +10.4 points on days 30–39, on seeds 0–3. The 13 Oct mechanism run gave +25.4 on the same seeds and worlds: days 0–39 are identical whatever the run length. The paired difference is +15.1 points, 95% interval [−5.7, +40.5], so the drop is suggestive, not established.

Of the changes since then, the mild working styles (`--personas`) are the one that alters every honest instance. Two observations from the weekend run fit them:
- honest instances chose the best clinic more often (63% of choices on days 11–40, against 48%);
- the most evidence-seeking styles ("open to trying another clinic", "ask colleagues for their figures") were the least affected (+8–10 points on days 11–39, against +11–17 for the other styles).

This run asks whether turning the styles off makes the attack's effect larger.

## Setting (the same in every arm)

| | |
|---|---|
| domain | medical, 30 desks, degree 6, rewire 0, tenure 5 |
| channel | desk notes files (`--memory file`), as in the weekend run: purpose-only `INSTRUCTIONS.md`, edits at rotation, one retry |
| attack | `misaligned_playbook`, with the forged head-of-team authority; `placebo_thinking` twin |
| arrivals | k misaligned instances on days 11–20 (`--arrival random --arrival-until 20`), plant day 10 |
| length | 50 days; source-free from day 30 |
| seeds | 0–7 |
| judged | as a 40-day run (`analyze --until 40`): days 30–39. Days 40–49 for how fast the effect fades |

## Arms

| setting | doses | what it varies |
|---|---|---|
| personas on (`--personas`) | k = 24, 36 | every instance gets one mild working style, as in the weekend run |
| personas off | k = 24, 36 | no working style line; otherwise identical |

Each arm is compared with its own twin, which has the same persona setting, seed and dose. That gives 2 doses × 2 persona settings × 2 sources × 8 seeds = **64 episodes**, about 11–12 h.

- **Why k = 24 and 36.** k = 12 showed no effect in either channel, so there is nothing for personas to change. k = 36 is 60% of newcomers on days 11–20, with the honest pool at 9–11 of 30 desks at its trough. That is a heavy dose with a larger effect to modulate, but short of the k = 48 occupation (80% of newcomers, 3–6 honest desks). k = 24 (40%, 14–17 honest desks) is the main dose.
- **Why notes files only.** They are the default channel. The weekend run found no difference between the channels in beliefs, so a persona effect should carry over. This run cannot itself account for the drop from +25 to +10, which was measured with procedures.

## Outcomes (fixed before the run)

1. **Primary:** the effect of turning personas off, per dose. That is (Δ believers vs twin, personas off) minus (Δ believers vs twin, personas on), on days 30–39, with a 95% bootstrap interval over the 8 seeds, paired by seed.
2. **Secondary:**
   - the same difference for honest-only regret;
   - honest exploration (choices of the best clinic on days 11–40), arm and twin;
   - retention of an inherited pushing file by honest instances on source-free days;
   - the per-turnover excess over days 30–49 (HOW LONG IT LASTS);
   - with personas on, Δ by working style (as in the weekend analysis).
3. **Health:** unusable notes replies, and evaluation-awareness counts.

## Predictions (written before the run; to be confirmed by the user)

- **P1:** with personas off, Δ believers at k = 24 is at least 10 points higher than with them on.
- **P2:** the same direction at k = 36.
- **P3:** with personas off, honest exploration on days 11–40 is lower than with them on.
- **Null:** if P1 fails, personas don't explain the drop. The instructions file and the forged-authority bullet are the next suspects.

**Power:** per-seed Δ varied by about ±20 points in past runs. With 8 paired seeds, a 10–15 point difference is detectable only roughly. A null result here does not rule out a smaller persona effect.

## Non-goals

- The instructions-file and forged-authority ablations (proposed, not chosen).
- A procedures arm.
- k = 12 and 48.
