# 08 — How busy will the gym be tomorrow? (and why your score is lying)

Your gym publishes how many people came in each day. You want to predict
tomorrow, so you know whether to go at 6pm or give up and go at 6am.

You build a model, cross-validate it, and get a great score. Then you use it
for a month and it is noticeably worse than the score promised.

**Nothing is wrong with the model. Your validation was leaking.**

---

## What leakage is

Information from the test set sneaks into training, so the model is graded on
questions it has effectively already seen the answers to. The score goes up.
Real-world performance does not.

It is the most common way a model looks good in a notebook and fails in
production, and it produces no error, no warning, nothing. Just a number
that is too good.

## The two leaks in this exercise

**Leak 1 — the wrong split for time.** Random k-fold shuffles the days, so the
model trains on Tuesday and Thursday, then is tested on Wednesday. In real
life you never get to see Thursday before predicting Wednesday. The fix is a
time-ordered split: always train on the past, test on the future.

**Leak 2 — preprocessing before splitting.** You pick the "most useful"
features using ALL the data, then cross-validate. The selection step already
peeked at the test folds. The demo uses 5,000 columns of pure random noise
and random labels — and still gets a high score. The fix: every step that
learns from data happens inside the fold.

## The solution

`solution.py` runs the experiment with `KFold`, `TimeSeriesSplit`,
`Pipeline` and `cross_val_score`.

## Why this matters for your projects

- **RAG project** — your golden set is an evaluation. If you tune prompts
  while looking at the same questions you report on, you have leaked.
- **RecSys project** — temporal splits are mandatory. Random splits on
  interaction data are the single most common credibility mistake.
- **Every interview** — "how did you validate it?" is asked about every
  project. This exercise is your answer.

## Time: 60 minutes
