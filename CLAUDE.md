# CLAUDE.md

Guidance for coding agents (and humans) working on this repository. Project rules first, then two general rule sets
this project follows: Andrej Karpathy's coding guidelines and Ponytail's "simplest thing that works" ladder.

## Project rules (SHL grammar scoring)

- **Validation before the leaderboard.** Every change is judged on speaker-grouped out-of-fold predictions at
  test-like lengths (the 4 crops), with the leaderboard formula `0.6·RMSE + 0.4·(1 − Pearson)`. Accept a change only if
  a paired bootstrap over clips gives P(better) ≥ 0.9 *and* it is a change to how a model is trained or corrected.
  Blend-composition changes have failed on the public leaderboard ten times in a row (README §7).
- **Forecast before submitting.** Write the predicted public score down first (the averaging identity in README §8
  predicts mixes to within 0.0005).
- **Public-tuned vs honest.** Mix weights fitted to public scores are reported separately from the main model's score;
  the final picks keep one honestly validated model.
- **Licences.** Commercial-use licences only, no external training data.
- **Code style.** Numbered scripts in `src/`, one step each, a module docstring that says what it does, why, its output
  and usage. Shared settings live in `src/common.py`. New options keep the default behaviour byte-for-byte identical,
  so earlier results stay reproducible. Every script caches its output in `artifacts/`.
- **Log everything.** Each experiment, including failures, goes into `submissions/README.md` with its numbers.

## Karpathy guidelines

From [forrestchang/andrej-karpathy-skills](https://github.com/forrestchang/andrej-karpathy-skills) (MIT), based on
Andrej Karpathy's observations about LLM coding mistakes.

1. **Think before coding.** State assumptions explicitly; if uncertain, ask. If several interpretations exist, present
   them instead of picking silently. If a simpler approach exists, say so.
2. **Simplicity first.** Minimum code that solves the problem: no features, abstractions or configurability beyond what
   was asked, no error handling for impossible scenarios. If 200 lines could be 50, rewrite it.
3. **Surgical changes.** Touch only what the task needs; match the existing style; don't refactor what isn't broken.
   Remove only the orphans your own change created. Every changed line should trace to the request.
4. **Goal-driven execution.** Turn the task into verifiable goals (a check that fails before and passes after) and
   loop until verified. For multi-step work, state the plan as `step → verify: check`.

## Ponytail ladder

From [DietrichGebert/ponytail](https://github.com/DietrichGebert/ponytail) (MIT). Stop at the first rung that holds,
after reading the code the change touches:

1. Does this need to exist at all? (YAGNI)
2. Is it already in this codebase? Reuse it.
3. Does the standard library do it?
4. Does an already-installed dependency do it? Never add one for what a few lines can do.
5. Can it be one line?
6. Only then: the minimum code that works.

Mark deliberate shortcuts with a `ponytail:` comment that names the ceiling and the upgrade path. Never simplify away
validation of untrusted input or anything that prevents data loss, and leave one runnable check behind for
non-trivial logic.
