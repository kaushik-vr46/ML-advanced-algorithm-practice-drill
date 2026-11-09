"""Three years of daily gym footfall, plus a pure-noise dataset.

No downloads — everything is generated here.
"""
import numpy as np
import pandas as pd


def load_gym(n_days=1095, seed=8):
    """Daily footfall with the patterns a real gym has:

      - a slow upward trend (the gym is getting more popular)
      - a weekly cycle (busy Monday, dead Sunday)
      - a January spike (resolutions) that fades through the month
      - fewer people on warm days
      - AUTOCORRELATED noise — a busy day tends to follow a busy day

    That last one is what makes random splits leak. Neighbouring days are
    similar, so a model that has seen Tuesday and Thursday can nearly
    copy the answer for Wednesday.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    dow = dates.dayofweek.to_numpy()
    month = dates.month.to_numpy()

    trend = 120 + 0.08 * t
    weekly = np.array([22, 25, 20, 15, 4, -24, -34])[dow]
    january = np.where(month == 1, 45 * (1 - dates.day.to_numpy() / 31), 0)
    temp = 11 + 8 * np.sin(2 * np.pi * (t - 100) / 365) + rng.normal(0, 2, n_days)
    weather = -1.2 * (temp - 11)

    noise = np.zeros(n_days)
    for i in range(1, n_days):
        noise[i] = 0.9 * noise[i - 1] + rng.normal(0, 9)

    footfall = trend + weekly + january + weather + noise

    df = pd.DataFrame({
        "date": dates,
        "day_index": t,
        "day_of_week": dow,
        "month": month,
        "temp_c": temp.round(1),
        "footfall": footfall.round(0),
    })

    # Lag features use .shift(), so each row only sees PAST days.
    # On their own these are perfectly legitimate. The leak only appears
    # when the split lets the model train on days AFTER the test day.
    df["lag_1"] = df["footfall"].shift(1)
    df["lag_7"] = df["footfall"].shift(7)
    df["rolling_7"] = df["footfall"].shift(1).rolling(7).mean()

    return df.dropna().reset_index(drop=True)


# What most people build first: features about the calendar.
CALENDAR_FEATURES = ["day_index", "day_of_week", "month", "temp_c"]

# Adding what actually happened recently. Legitimate — every lag looks
# only backwards — and it is the real fix for this problem.
LAG_FEATURES = CALENDAR_FEATURES + ["lag_1", "lag_7", "rolling_7"]
GYM_TARGET = "footfall"


def load_noise(n_rows=100, n_cols=5000, seed=0):
    """Pure random noise with random labels.

    There is NOTHING to learn here. Any honest evaluation must report
    roughly 50% accuracy. Anything well above that is leakage.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_cols))
    y = rng.integers(0, 2, n_rows)
    return X, y
