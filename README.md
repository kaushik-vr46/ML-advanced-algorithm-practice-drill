# ML Practice: Tier 2 (evaluation, neural networks and attention)

Three exercises that follow on from Tier 1: honest model evaluation, neural networks in PyTorch, and sequence models from RNNs to attention.

## How this was made

I asked Claude to write a problem statement and a dataset for each exercise. I then typed out the solutions, and used Claude to understand the concepts and clear up my doubts along the way.

## Exercises

| # | Problem | Topic | Also covers |
|---|---|---|---|
| 08 | How busy will the gym be? | Evaluation leakage | Random k-fold vs time-series splits, preprocessing inside each fold |
| 09 | Reading handwritten meter readings | Neural networks in PyTorch | MLP, training loop, dropout and eval mode |
| 10 | Phone autocomplete | RNN to attention | Character-level RNN vs a tiny GPT, causal masking, long-range memory |

Each folder has:

- `problem.md`: the problem statement
- `data.py`: generates or loads the dataset
- `solution.py`: my solution
- `check.py`: checks the solution works (09 and 10)
- `plots/`: charts produced by the solution

## Running it

```bash
pip install numpy pandas scikit-learn matplotlib torch
cd 09_nn_handwritten_readings
python solution.py
python check.py
```

Everything trains on a laptop CPU in a minute or two.
