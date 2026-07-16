from abc import ABC, abstractmethod
import pandas as pd


class MetricHandler(ABC):

    @abstractmethod
    def aggregate(self, df:pd.DataFrame) -> pd.DataFrame:
        """Aggregate raw records for this metric type into daily rows.
        Input:  DataFrame filtered to this metric's type, with columns
                [type, startDate, endDate, value, duration].
        Output: DataFrame indexed by date (datetime, tz-naive, normalized)
                with one or more named columns.
        """


class Preprocessor:

    def __init__(self):
        self._handlers:dict[str,MetricHandler] = {}


    def register(self, metric_type:str, handler:MetricHandler):
        self._handlers[metric_type] = handler


    def process(self, df:pd.DataFrame) -> pd.DataFrame:
        daily_frames = []
        for metric_type, handler in self._handlers.items():
            subset = df[df["type"] == metric_type]
            if subset.empty:
                continue
            daily = handler.aggregate(subset)
            daily_frames.append(daily)
        if not daily_frames:
            return pd.DataFrame()
        result = daily_frames[0]
        for frame in daily_frames[1:]:
            result = result.join(frame, how="outer")
        result = result.sort_index()
        return result
