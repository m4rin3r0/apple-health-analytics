import pandas as pd
from apple_health_analytics.preprocessor import MetricHandler

SLEEP_STAGES = {"inbed", "asleepcore", "asleepdeep", "asleeprem", "awake", "asleepunspecified", "asleep"}


class SleepHandler(MetricHandler):

    def aggregate(self, df:pd.DataFrame) -> pd.DataFrame:
        daily = df.copy()
        daily = daily[daily["value"].isin(SLEEP_STAGES)]
        daily["date"] = daily["endDate"].dt.normalize().dt.tz_localize(None)
        daily["hours"] = daily["duration"] / 3600
        pivoted = daily.pivot_table(index="date", columns="value", values="hours", aggfunc="sum", fill_value=0)
        pivoted.columns = [f"sleep_{col}_hours" for col in pivoted.columns]
        if "sleep_inbed_hours" in pivoted.columns:
            pivoted["sleep_total_hours"] = pivoted.drop(columns=["sleep_inbed_hours"], errors="ignore").sum(axis=1)
        else:
            pivoted["sleep_total_hours"] = pivoted.sum(axis=1)
        return pivoted
