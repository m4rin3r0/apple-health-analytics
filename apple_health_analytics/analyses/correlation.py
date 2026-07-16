import pandas as pd
from apple_health_analytics.analyzer import Analysis


class CorrelationAnalysis(Analysis):
    """Finds pairs of metrics that move together — today and across days.

    Lag 0 answers "do these happen on the same day?" (e.g. more steps, more water).
    Lag 1+ answers "does today's behavior affect tomorrow?" (e.g. workout today,
    better sleep tonight or deeper sleep tomorrow). Uses Spearman correlation,
    which is robust to outliers and does not assume linear relationships.

    Pairs within the same metric group (e.g. sleep_deep vs sleep_rem) are skipped
    by default — they are almost always trivially correlated. Days where either
    metric has no data are excluded per pair, so metrics tracked over different
    periods do not produce artificial correlations.
    """

    def __init__(self, max_lag:int=1, min_abs_corr:float=0.3, min_days:int=30, include_within_group:bool=False):
        self.max_lag = max_lag
        self.min_abs_corr = min_abs_corr
        self.min_days = min_days
        self.include_within_group = include_within_group


    def run(self, df:pd.DataFrame) -> dict:
        features = self._usable_features(df)
        pairs = []
        for lag in range(self.max_lag + 1):
            pairs.extend(self._correlate_at_lag(features, lag))
        pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)
        return {"pairs": pairs, "n_days": len(df), "features": features.columns.tolist()}


    def _usable_features(self, df:pd.DataFrame) -> pd.DataFrame:
        numeric = df.select_dtypes(include="number")
        non_constant = numeric.loc[:, numeric.std() > 0]
        return non_constant


    def _same_group(self, feature_a:str, feature_b:str) -> bool:
        return feature_a.split("_")[0] == feature_b.split("_")[0]


    def _correlate_at_lag(self, features:pd.DataFrame, lag:int) -> list[dict]:
        shifted = features.shift(lag)
        results = []
        columns = features.columns.tolist()
        for i, cause in enumerate(columns):
            for j, effect in enumerate(columns):
                if lag == 0 and j <= i:
                    continue
                if lag > 0 and cause == effect:
                    continue
                if not self.include_within_group and self._same_group(cause, effect):
                    continue
                valid = pd.concat([shifted[cause], features[effect]], axis=1).dropna()
                if len(valid) < self.min_days:
                    continue
                corr = valid.iloc[:, 0].corr(valid.iloc[:, 1], method="spearman")
                if pd.isna(corr) or abs(corr) < self.min_abs_corr:
                    continue
                results.append({"cause": cause, "effect": effect, "lag": lag, "correlation": round(float(corr), 3), "n_days": len(valid)})
        return results
