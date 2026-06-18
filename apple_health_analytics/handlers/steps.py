import pandas as pd
from apple_health_analytics.preprocessor import MetricHandler


class StepsHandler(MetricHandler):

    def aggregate(self, df:pd.DataFrame) -> pd.DataFrame:
        daily = df.copy()
        daily["date"] = daily["startDate"].dt.normalize().dt.tz_localize(None)
        daily = daily.groupby("date").agg(steps=("value", "sum"))
        return daily
