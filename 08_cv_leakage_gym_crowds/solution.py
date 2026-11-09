from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from data import load_gym, load_noise, CALENDAR_FEATURES, LAG_FEATURES, GYM_TARGET

def three_errors(X, y):

    model = RandomForestRegressor(n_estimators = 150, random_state = 0, n_jobs = -1)

    random_splitter = KFold(n_splits = 5, shuffle = True, random_state = 0)
    random_scores = cross_val_score(model, X, y, cv=random_splitter, 
                                    scoring = "neg_mean_absolute_error")
    random_cv_error = -random_scores.mean()

    time_splitter = TimeSeriesSplit(n_splits = 5, gap = 7)
    time_scores = cross_val_score(model, X, y, cv = time_splitter, 
                                  scoring = "neg_mean_absolute_error")
    time_cv_error = -time_scores.mean()

    cut = int(0.8 * len(y))
    model.fit(X[:cut], y[:cut])
    predictions = model.predict(X[cut:])
    future_error = mean_absolute_error(y[cut:], predictions)

    return random_cv_error, time_cv_error, future_error

def main():
    df = load_gym()
    y = df[GYM_TARGET].to_numpy()

    X_calendar = df[CALENDAR_FEATURES].to_numpy()
    X_lags = df[LAG_FEATURES].to_numpy()

    random_A, time_A, future_A = three_errors(X_calendar, y)
    random_B, time_B, future_B = three_errors(X_lags, y)

    print("Average mistake in people per day")
    print()
    print("Model A - Calendar only:")
    print("random CV:", round(random_A, 1), "time CV:", round(time_A, 1), "real future:", round(future_A, 1))
    print()
    print("Model B - Calendar + Lags:")
    print("random CV:", round(random_B, 1), "time CV:", round(time_B, 1), "real future:", round(future_B, 1))
    print()
    print("Only time-series split tracks what really happens in the future")
    print()

    # Step 3: compare the wrong and right approaches.
    X_noise, y_noise = load_noise()
    splitter = KFold(n_splits=5, shuffle=True, random_state=0)

    # Wrong approach: selector sees every row before CV.
    selector = SelectKBest(f_classif, k=20)
    X_selected = selector.fit_transform(X_noise, y_noise)
    model = LogisticRegression(max_iter=1000)
    scores = cross_val_score(model, X_selected, y_noise, cv=splitter)
    wrong_score = scores.mean()

    # Right approach: cross_val_score refits the selector on each training fold.
    pipeline = Pipeline([
        ("select", SelectKBest(f_classif, k=20)),
        ("model", LogisticRegression(max_iter=1000)),
    ])

    scores_right = cross_val_score(pipeline, X_noise, y_noise, cv=splitter)
    right_score = scores_right.mean()

    print("Selector outside the pipeline:", round(wrong_score * 100), "%")
    print("Selector inside the pipeline:", round(right_score * 100), "%")

    # Step 4: Plot the points for gym results

    labels = ["random CV", "time CV", "real future"]
    model_A_values = [random_A, time_A, future_A]
    model_B_values = [random_B, time_B, future_B]

    positions = np.arange(3)
    bar_width = 0.35

    plt.figure(figsize=(8, 4.5))
    plt.bar(positions - bar_width / 2, model_A_values, bar_width, label="Model A: calendar")
    plt.bar(positions + bar_width / 2, model_B_values, bar_width, label="Model B: + lags")
    plt.xticks(positions, labels)
    plt.ylabel("Mistake (people per day)")
    plt.title("Which evaluation told the truth?")
    plt.legend()
    plt.tight_layout()
    plot_path = Path(__file__).resolve().parent / "plots" / "leakage.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=110)
    print()
    print("Saved plot to plots/leakage.png")

if __name__ == "__main__":
    main()






