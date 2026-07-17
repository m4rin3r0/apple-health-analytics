import pandas as pd
from apple_health_analytics.preprocessor import MetricHandler

SLEEP_STAGES = {"inbed", "asleepcore", "asleepdeep", "asleeprem", "awake", "asleepunspecified", "asleep"}
DETAILED_STAGES = ["asleepcore", "asleepdeep", "asleeprem", "awake"]
STAGE_INDICATORS = ["asleepcore", "asleepdeep", "asleeprem"]
ASLEEP_VALUES = ["asleepcore", "asleepdeep", "asleeprem", "asleepunspecified", "asleep"]


class SleepHandler(MetricHandler):
    """Aggregates sleep records into daily hours per sleep stage.

    Detailed stages (core/deep/rem/awake) only exist on devices that measure
    them — a night from an older watch has none. On days without any detailed
    stage measurement those columns stay NaN ("not measured") instead of 0
    ("measured zero"), so correlations do not pick up tracking eras as fake
    patterns. A night is assigned to the day the sleep ended (wake-up day).
    """

    def aggregate(self, df:pd.DataFrame) -> pd.DataFrame:
        daily = df.copy()
        daily = daily[daily["value"].isin(SLEEP_STAGES)]
        daily["date"] = daily["endDate"].dt.normalize().dt.tz_localize(None)
        daily["hours"] = daily["duration"] / 3600
        pivoted = daily.pivot_table(index="date", columns="value", values="hours", aggfunc="sum")
        pivoted = self._zero_fill_measured_stages(pivoted)
        asleep_cols = [col for col in ASLEEP_VALUES if col in pivoted.columns]
        total = pivoted[asleep_cols].sum(axis=1, min_count=1)
        pivoted.columns = [f"sleep_{col}_hours" for col in pivoted.columns]
        pivoted["sleep_total_hours"] = total
        return pivoted


    def _zero_fill_measured_stages(self, pivoted:pd.DataFrame) -> pd.DataFrame:
        indicators = [col for col in STAGE_INDICATORS if col in pivoted.columns]
        present = [col for col in DETAILED_STAGES if col in pivoted.columns]
        if not indicators:
            return pivoted
        has_stages = pivoted[indicators].notna().any(axis=1)
        pivoted.loc[has_stages, present] = pivoted.loc[has_stages, present].fillna(0)
        return pivoted
