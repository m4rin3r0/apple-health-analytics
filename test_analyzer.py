import numpy as np
import pandas as pd
from apple_health_analytics.analyzer import Analyzer
from apple_health_analytics.analyses.correlation import CorrelationAnalysis
from apple_health_analytics.analyses.weekday import WeekdayPatternAnalysis


def build_synthetic_daily_data(n_days:int=365) -> pd.DataFrame:
    """Builds a year of fake daily health data with known hidden patterns:
    - workout days are followed by more deep sleep the same night (lag 0 on endDate logic)
    - more steps on days with workouts
    - long weekend sleep (Saturday/Sunday)
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    workout_hours = rng.choice([0, 0, 0, 1.0, 1.5], size=n_days) + rng.normal(0, 0.1, n_days).clip(0)
    steps = 4000 + workout_hours * 5000 + rng.normal(0, 1500, n_days).clip(-3000)
    sleep_deep = 1.0 + workout_hours * 0.5 + rng.normal(0, 0.3, n_days).clip(-0.8)
    is_weekend = pd.Series(dates).dt.dayofweek.isin([5, 6]).to_numpy()
    sleep_total = 6.5 + is_weekend * 1.5 + rng.normal(0, 0.5, n_days)
    hydration = 1500 + steps * 0.1 + rng.normal(0, 300, n_days)
    return pd.DataFrame({"steps": steps, "workout_hours": workout_hours, "sleep_deep_hours": sleep_deep, "sleep_total_hours": sleep_total, "hydration_ml": hydration}, index=dates)


def main():
    df = build_synthetic_daily_data()
    print(f"=== Synthetic data: {len(df)} days, {len(df.columns)} features ===")
    analyzer = Analyzer()
    analyzer.register("correlation", CorrelationAnalysis(max_lag=1, min_abs_corr=0.3))
    analyzer.register("weekday", WeekdayPatternAnalysis(min_deviation=0.1))
    results = analyzer.run_all(df)
    print("\n=== Correlations (should find: workout->steps, workout->deep sleep, steps->hydration) ===")
    for pair in results["correlation"]["pairs"][:10]:
        arrow = "same day" if pair["lag"] == 0 else f"{pair['lag']} day(s) later"
        print(f"  {pair['cause']} -> {pair['effect']} ({arrow}): r={pair['correlation']}")
    print("\n=== Weekday patterns (should find: longer sleep on weekends) ===")
    for pattern in results["weekday"]["notable"]:
        sign = "+" if pattern["deviation_pct"] > 0 else ""
        print(f"  {pattern['feature']}: {sign}{pattern['deviation_pct']}% on {pattern['weekday']}")


if __name__ == "__main__":
    main()
