import pandas as pd
from apple_health_analytics.preprocessor import MetricHandler


class WorkoutHandler(MetricHandler):

    def aggregate(self, df:pd.DataFrame) -> pd.DataFrame:
        daily = df.copy()
        daily["date"] = daily["startDate"].dt.normalize().dt.tz_localize(None)
        daily["hours"] = daily["duration"] / 3600
        by_date = daily.groupby("date").agg(
            workout_count=("value", "count"),
            workout_hours=("hours", "sum"))
        type_pivot = daily.pivot_table(index="date", columns="value", values="hours", aggfunc="sum", fill_value=0)
        type_pivot.columns = [f"workout_{col}_hours" for col in type_pivot.columns]
        result = by_date.join(type_pivot, how="outer").fillna(0)
        return result
