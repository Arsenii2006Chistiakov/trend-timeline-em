"""
total_volume_em.py

Single-Gaussian "total volume" estimator for timestamped videos.

This module fits one Gaussian release curve while allowing the final total
volume ``Z`` to be larger than the number of observed timestamps. The model is
meant for the case where observed timestamps are treated as partial cumulative
evidence from a larger latent catalog.

The fitting routine is a lightweight generalized-EM style alternation:

1. E-step:
   Given the current Gaussian CDF ``F(t; mu, sigma)``, infer the latent total
   volume from the implied relation ``E[X_i] ~= Z * F(t_i)`` where
   ``X_i = i`` for sorted timestamps.
2. M-step:
   Given the current total volume estimate ``Z``, refit ``mu`` and ``sigma``
   from the implied Gaussian quantiles of the cumulative ranks.

This is intentionally simple and deterministic and uses only the Python
standard library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from .analyze_dates import day_number_to_iso_date, unix_to_day_number


EPS = 1e-9


def normal_cdf(x: float, mean: float, std: float) -> float:
    """
    Normal CDF using ``erf``.
    """
    std = max(float(std), 1e-9)
    z = (x - mean) / (std * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def inverse_normal_cdf(p: float) -> float:
    """
    Approximate inverse standard normal CDF.

    Uses Peter John Acklam's rational approximation.
    """
    p = min(max(float(p), EPS), 1.0 - EPS)

    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )

    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)

    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    )


def _mean(values: List[float]) -> float:
    return sum(values) / float(len(values)) if values else math.nan


def _std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) * (v - m) for v in values) / float(len(values)))


def _weighted_mean(values: List[float], weights: List[float]) -> float:
    total_weight = sum(weights)
    if total_weight <= 0.0:
        return _mean(values)
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def _fit_gaussian_from_total_volume(
    day_values: List[float],
    total_volume: int,
    min_std_days: float,
) -> tuple[float, float]:
    """
    Refit ``mu`` and ``sigma`` from cumulative Gaussian quantiles.
    """
    n = len(day_values)
    if n == 1:
        return day_values[0], max(min_std_days, 1.0)

    probs = [
        min(max(i / float(total_volume + 1), EPS), 1.0 - EPS)
        for i in range(1, n + 1)
    ]
    quantiles = [inverse_normal_cdf(p) for p in probs]

    q_mean = _mean(quantiles)
    t_mean = _mean(day_values)
    q_var = sum((q - q_mean) * (q - q_mean) for q in quantiles)

    if q_var <= 0.0:
        return t_mean, max(_std(day_values), min_std_days)

    cov = sum((q - q_mean) * (t - t_mean) for q, t in zip(quantiles, day_values))
    sigma = max(cov / q_var, min_std_days)
    mu = t_mean - sigma * q_mean
    return mu, sigma


def _estimate_total_volume(
    day_values: List[float],
    mean_day: float,
    std_days: float,
    observed_count: int,
    max_total_volume: int,
) -> int:
    """
    Infer a stable total volume from cumulative coverage ratios.
    """
    implied_totals: List[float] = []
    weights: List[float] = []
    tail_count = max(3, int(math.ceil(observed_count * 0.25)))
    start_index = max(1, observed_count - tail_count + 1)

    for i, t in enumerate(day_values, start=1):
        if i < start_index:
            continue
        cdf = min(max(normal_cdf(t, mean_day, std_days), EPS), 1.0 - EPS)
        implied_totals.append(i / cdf)
        weights.append(float(i))

    if not implied_totals:
        return observed_count

    total_estimate = _weighted_mean(implied_totals, weights)
    total_estimate = max(float(observed_count), min(float(max_total_volume), total_estimate))
    return max(observed_count, int(round(total_estimate)))


def _pseudo_log_likelihood(
    day_values: List[float],
    mean_day: float,
    std_days: float,
    total_volume: int,
) -> float:
    """
    Pseudo log-likelihood for cumulative counts under a Binomial/CDF view.
    """
    ll = 0.0
    for i, t in enumerate(day_values, start=1):
        cdf = min(max(normal_cdf(t, mean_day, std_days), EPS), 1.0 - EPS)
        ll += math.lgamma(total_volume + 1)
        ll -= math.lgamma(i + 1)
        ll -= math.lgamma(total_volume - i + 1)
        ll += i * math.log(cdf)
        ll += (total_volume - i) * math.log(1.0 - cdf)
    return ll


@dataclass
class TotalVolumeEMResult:
    observed_volume: int
    estimated_total_volume: int
    estimated_unseen_volume: int
    mean_day: float
    std_days: float
    log_likelihood: float
    iterations: int
    converged: bool


def fit_total_volume_gaussian_em(
    unix_timestamps: Iterable[float],
    max_iter: int = 100,
    tol: float = 1e-3,
    min_std_days: float = 2.0,
    max_total_volume_ratio: float = 25.0,
) -> Dict[str, Any]:
    """
    Fit one Gaussian release curve with a latent total volume ``Z``.

    Parameters
    ----------
    unix_timestamps : iterable of float
        Observed timestamps in Unix seconds.
    max_iter : int, optional
        Maximum generalized-EM iterations.
    tol : float, optional
        Convergence tolerance on ``mu``/``sigma`` changes.
    min_std_days : float, optional
        Lower bound for the fitted standard deviation in days.
    max_total_volume_ratio : float, optional
        Hard cap on ``estimated_total_volume`` as a multiple of the number of
        observed timestamps, to prevent unstable tail estimates from exploding.

    Returns
    -------
    dict
        {
          "gaussian_components": [
            {
              "mean_date": "YYYY-MM-DD",
              "start_date": "YYYY-MM-DD",
              "end_date": "YYYY-MM-DD",
              "weight": float,
              "std_days": float,
            }
          ],
          "start_date": "YYYY-MM-DD" | None,
          "end_date": "YYYY-MM-DD" | None,
          "observed_volume": int,
          "estimated_total_volume": int,
          "estimated_unseen_volume": int,
          "mean_date": "YYYY-MM-DD" | None,
          "std_days": float | None,
          "mean_day": float | None,
          "log_likelihood": float | None,
          "iterations": int,
          "converged": bool,
          "curve": [
            {
              "date": "YYYY-MM-DD",
              "observed_cumulative": int,
              "fitted_cdf": float,
              "expected_cumulative": float,
            },
            ...
          ],
        }
    """
    day_values: List[float] = []
    for ts in unix_timestamps:
        try:
            value = float(ts)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        day_values.append(unix_to_day_number(value))

    day_values.sort()
    observed_count = len(day_values)
    if observed_count == 0:
        return {
            "gaussian_components": [],
            "start_date": None,
            "end_date": None,
            "observed_volume": 0,
            "estimated_total_volume": 0,
            "estimated_unseen_volume": 0,
            "mean_date": None,
            "std_days": None,
            "mean_day": None,
            "log_likelihood": None,
            "iterations": 0,
            "converged": True,
            "curve": [],
        }

    mean_day = _mean(day_values)
    std_days = max(_std(day_values), min_std_days)
    total_volume = observed_count
    max_total_volume = max(observed_count, int(math.ceil(observed_count * max_total_volume_ratio)))

    converged = False
    iterations = 0
    for iteration in range(1, max_iter + 1):
        prev_mean = mean_day
        prev_std = std_days
        prev_total = total_volume

        total_volume = _estimate_total_volume(
            day_values,
            mean_day,
            std_days,
            observed_count=observed_count,
            max_total_volume=max_total_volume,
        )
        mean_day, std_days = _fit_gaussian_from_total_volume(
            day_values,
            total_volume=total_volume,
            min_std_days=min_std_days,
        )

        iterations = iteration
        mean_delta = abs(mean_day - prev_mean)
        std_delta = abs(std_days - prev_std)
        if total_volume == prev_total and mean_delta <= tol and std_delta <= tol:
            converged = True
            break

    log_likelihood = _pseudo_log_likelihood(day_values, mean_day, std_days, total_volume)

    curve: List[Dict[str, Any]] = []
    for i, day_num in enumerate(day_values, start=1):
        cdf = min(max(normal_cdf(day_num, mean_day, std_days), 0.0), 1.0)
        curve.append(
            {
                "date": day_number_to_iso_date(day_num),
                "observed_cumulative": i,
                "fitted_cdf": float(cdf),
                "expected_cumulative": float(total_volume * cdf),
            }
        )

    result = TotalVolumeEMResult(
        observed_volume=observed_count,
        estimated_total_volume=total_volume,
        estimated_unseen_volume=max(0, total_volume - observed_count),
        mean_day=mean_day,
        std_days=std_days,
        log_likelihood=log_likelihood,
        iterations=iterations,
        converged=converged,
    )
    mean_date = day_number_to_iso_date(result.mean_day)
    start_date = day_number_to_iso_date(result.mean_day - 2.0 * result.std_days)
    end_date = day_number_to_iso_date(result.mean_day + 2.0 * result.std_days)
    component = {
        "mean_date": mean_date,
        "start_date": start_date,
        "end_date": end_date,
        "weight": 1.0,
        "std_days": float(result.std_days),
    }

    return {
        "gaussian_components": [component],
        "start_date": start_date,
        "end_date": end_date,
        "observed_volume": result.observed_volume,
        "estimated_total_volume": result.estimated_total_volume,
        "estimated_unseen_volume": result.estimated_unseen_volume,
        "mean_date": mean_date,
        "std_days": float(result.std_days),
        "mean_day": float(result.mean_day),
        "log_likelihood": float(result.log_likelihood),
        "iterations": result.iterations,
        "converged": result.converged,
        "curve": curve,
    }


__all__ = [
    "TotalVolumeEMResult",
    "fit_total_volume_gaussian_em",
    "inverse_normal_cdf",
    "normal_cdf",
]
