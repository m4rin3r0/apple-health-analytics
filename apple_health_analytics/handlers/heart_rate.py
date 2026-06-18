import pandas as pd
from apple_health_analytics.preprocessor import MetricHandler


class HeartRateHandler(MetricHandler):

    def aggregate(self, df:pd.DataFrame) -> pd.DataFrame:
        daily = df.copy()
        daily["date"] = daily["startDate"].dt.normalize().dt.tz_localize(None)
        daily = daily.groupby("date").agg(
            heart_rate_mean=("value", "mean"),
            heart_rate_min=("value", "min"),
            heart_rate_max=("value", "max"))
        return daily
