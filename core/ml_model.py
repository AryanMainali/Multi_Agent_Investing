"""Walk-forward Logistic Regression direction model with embargo gap (no lookahead).

Incorporates the team's machine-learning Strategy design:
- Features: 1d/5d/20d returns, 20d volatility, distance from 50d MA, 20d volume z-score.
- Walk-forward evaluation: trains strictly on past labeled rows separated by an embargo
  gap of `horizon_days` so training labels never peek into the held-out test window.
- Measures held-out `backtest_accuracy` and predicts `horizon_days` forward probability.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

ML_FEATURES = ['ret_1', 'ret_5', 'ret_20', 'vol_20', 'dist_ma50', 'volume_z']


@dataclass
class StandardizedLogisticModel:
    """StandardScaler + L2-regularized Logistic Regression in exact vectorized NumPy."""
    mean_: np.ndarray
    scale_: np.ndarray
    weights_: np.ndarray  # shape (d + 1,), index 0 is intercept
    constant_prob_: float | None = None

    def predict_proba_up(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        if self.constant_prob_ is not None:
            return np.full(X_arr.shape[0], self.constant_prob_, dtype=float)
        Z = (X_arr - self.mean_) / self.scale_
        Z_aug = np.column_stack([np.ones(Z.shape[0], dtype=float), Z])
        logits = np.clip(Z_aug @ self.weights_, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba_up(X) >= 0.5).astype(int)


def fit_logistic_classifier(
    X: np.ndarray,
    y: np.ndarray,
    l2_penalty: float = 1.0,
    max_iter: int = 50,
    tol: float = 1e-7,
) -> StandardizedLogisticModel:
    """Fit standardized L2-regularized binary logistic regression via Newton-IRLS."""
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    n_samples, n_features = X_arr.shape

    mean = X_arr.mean(axis=0)
    scale = X_arr.std(axis=0, ddof=0)
    scale = np.where(scale < 1e-12, 1.0, scale)

    unique_classes = np.unique(y_arr)
    if len(unique_classes) < 2:
        # Laplace-smoothed empirical rate when history is purely monotonic
        smoothed = float((y_arr.sum() + 1.0) / (n_samples + 2.0))
        return StandardizedLogisticModel(
            mean_=mean,
            scale_=scale,
            weights_=np.zeros(n_features + 1, dtype=float),
            constant_prob_=smoothed,
        )

    Z = (X_arr - mean) / scale
    Z_aug = np.column_stack([np.ones(n_samples, dtype=float), Z])
    reg_vec = np.r_[0.0, np.full(n_features, float(l2_penalty))]

    w = np.zeros(n_features + 1, dtype=float)
    for _ in range(max_iter):
        logits = np.clip(Z_aug @ w, -35.0, 35.0)
        p = 1.0 / (1.0 + np.exp(-logits))
        grad = Z_aug.T @ (p - y_arr) + reg_vec * w
        w_diag = np.clip(p * (1.0 - p), 1e-8, None)
        hess = (Z_aug.T * w_diag) @ Z_aug + np.diag(reg_vec)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        w = w - step
        if float(np.max(np.abs(step))) < tol:
            break

    return StandardizedLogisticModel(mean_=mean, scale_=scale, weights_=w, constant_prob_=None)


def build_ml_features(df: pd.DataFrame, horizon_days: int = 5) -> pd.DataFrame:
    """Compute past-only price/volume features and optional forward target for training."""
    work = df.copy()
    close_col = 'Close' if 'Close' in work.columns else 'close'
    vol_col = 'Volume' if 'Volume' in work.columns else 'volume'

    close = work[close_col].astype(float)
    volume = work[vol_col].astype(float)

    out = pd.DataFrame(index=work.index)
    out['Close'] = close
    out['ret_1'] = close.pct_change(1)
    out['ret_5'] = close.pct_change(5)
    out['ret_20'] = close.pct_change(20)
    out['vol_20'] = out['ret_1'].rolling(20, min_periods=20).std(ddof=0)
    ma_50 = close.rolling(50, min_periods=50).mean()
    out['dist_ma50'] = close / ma_50 - 1.0

    vol_mean = volume.rolling(20, min_periods=20).mean()
    vol_std = volume.rolling(20, min_periods=20).std(ddof=0).replace(0.0, np.nan)
    out['volume_z'] = ((volume - vol_mean) / vol_std).fillna(0.0)

    future = close.shift(-int(horizon_days))
    out['target'] = np.where(future.isna(), np.nan, (future > close).astype(float))
    return out


def predict_walk_forward(
    df: pd.DataFrame,
    horizon_days: int = 5,
    min_history: int = 60,
    test_window: int = 30,
) -> dict | None:
    """Train walk-forward logistic regression with embargo gap and score the latest bar."""
    horizon = max(1, int(horizon_days))
    feats = build_ml_features(df, horizon_days=horizon).dropna(subset=ML_FEATURES)
    if feats.empty:
        return None

    labeled = feats.dropna(subset=['target'])
    # Adapt windows gracefully when shorter lookback windows (e.g. 6mo / 180 bars) are passed
    avail = len(labeled)
    if avail < 45 + horizon:
        return None

    eff_test = min(int(test_window), max(15, avail // 4))
    eff_min_train = max(30, min(int(min_history), avail - eff_test - horizon))
    if avail < eff_min_train + eff_test + horizon:
        return None

    # Embargo gap of `horizon` rows prevents training targets from peeking into test period
    train = labeled.iloc[: -(eff_test + horizon)]
    test = labeled.iloc[-eff_test:]

    eval_model = fit_logistic_classifier(train[ML_FEATURES].values, train['target'].values)
    preds = eval_model.predict(test[ML_FEATURES].values)
    accuracy = float((preds == test['target'].values.astype(int)).mean())

    # Refit on all past-settled labeled rows (strictly >= horizon bars before latest close)
    live_model = fit_logistic_classifier(labeled[ML_FEATURES].values, labeled['target'].values)
    latest = feats.iloc[[-1]]
    p_up = float(live_model.predict_proba_up(latest[ML_FEATURES].values)[0])
    direction = 'up' if p_up >= 0.5 else 'down'
    confidence = max(p_up, 1.0 - p_up)

    as_of = str(latest.index[0].date()) if hasattr(latest.index[0], 'date') else str(latest.index[0])
    feature_snapshot = {k: round(float(latest[k].iloc[0]), 5) for k in ML_FEATURES}
    feature_snapshot['p_up'] = round(p_up, 4)

    return {
        'available': True,
        'as_of': as_of,
        'direction': direction,
        'p_up': round(p_up, 4),
        'probability': round(confidence, 4),
        'backtest_accuracy': round(accuracy, 4),
        'horizon_days': horizon,
        'train_samples': int(len(labeled)),
        'test_window': int(eff_test),
        'entry_ref_price': round(float(latest['Close'].iloc[0]), 4),
        'features': feature_snapshot,
    }


def generate_walk_forward_signal_series(
    prices: pd.DataFrame,
    horizon_days: int = 5,
    min_train_rows: int = 55,
    retrain_every: int = 10,
    entry_prob: float = 0.54,
    exit_prob: float = 0.46,
) -> pd.Series:
    """Compute strictly past-only p_up series across history for backtesting without lookahead.

    At bar index `t`, only rows `<= t - horizon_days` have settled `horizon_days`-ahead outcomes!
    """
    horizon = max(1, int(horizon_days))
    feats = build_ml_features(prices, horizon_days=horizon)
    valid_mask = feats[ML_FEATURES].notna().all(axis=1).values
    X_all = feats[ML_FEATURES].values
    y_all = feats['target'].values
    n = len(feats)

    p_up_series = np.full(n, np.nan, dtype=float)
    model: StandardizedLogisticModel | None = None
    last_fit_idx = -999_999

    for t in range(n):
        if not valid_mask[t]:
            continue
        # Latest row whose `horizon`-day forward return is fully known at bar `t` close is `t - horizon`
        settled_end = t - horizon + 1
        if settled_end <= 0:
            continue

        settled_valid = valid_mask[:settled_end] & ~np.isnan(y_all[:settled_end])
        if int(settled_valid.sum()) < min_train_rows:
            continue

        if model is None or (t - last_fit_idx) >= retrain_every:
            model = fit_logistic_classifier(X_all[:settled_end][settled_valid], y_all[:settled_end][settled_valid])
            last_fit_idx = t

        p_up_series[t] = float(model.predict_proba_up(X_all[t : t + 1])[0])

    return pd.Series(p_up_series, index=prices.index, name='ML_P_UP')
