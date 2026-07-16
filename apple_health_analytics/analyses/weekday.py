import pandas as pd
from apple_health_analytics.analyzer import Analysis

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class WeekdayPatternAnalysis(Analysis):
    """Finds systematic weekday patterns — which metrics differ by day of week.

    For every feature the mean per weekday is compared against the overall mean.
    The deviation is expressed as a percentage, so "sleep_total_hours: +12% on
    Saturday" reads directly. Sparse features (data on fewer than min_coverage
    of all days) are skipped — rare events produce huge but meaningless
    percentage deviations. Only features whose strongest weekday deviation
    exceeds min_deviation are reported as notable.
    """

    def __init__(self, min_deviation:float=0.15, min_coverage:float=0.2):
        self.min_deviation = min_deviation
        self.min_coverage = min_coverage


    def run(self, df:pd.DataFrame) -> dict:
        numeric = df.select_dtypes(include="number")
        coverage = (numeric.notna() & (numeric != 0)).mean()
        covered = numeric.loc[:, coverage >= self.min_coverage]
        by_weekday = covered.groupby(covered.index.day_name()).mean().reindex(WEEKDAYS)
        overall_mean = covered.mean()
        deviation = (by_weekday - overall_mean) / overall_mean.replace(0, pd.NA)
        notable = self._notable_patterns(deviation)
        return {"weekday_means": by_weekday, "deviation": deviation, "notable": notable, "skipped_sparse": coverage[coverage < self.min_coverage].index.tolist()}


    def _notable_patterns(self, deviation:pd.DataFrame) -> list[dict]:
        patterns = []
        for feature in deviation.columns:
            series = deviation[feature].dropna()
            if series.empty:
                continue
            strongest_day = series.abs().idxmax()
            strength = float(series[strongest_day])
            if abs(strength) < self.min_deviation:
                continue
            patterns.append({"feature": feature, "weekday": strongest_day, "deviation_pct": round(strength * 100, 1)})
        patterns.sort(key=lambda p: abs(p["deviation_pct"]), reverse=True)
        return patterns
