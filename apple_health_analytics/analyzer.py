from abc import ABC, abstractmethod
import pandas as pd


class Analysis(ABC):

    @abstractmethod
    def run(self, df:pd.DataFrame) -> dict:
        """Run this analysis on a daily feature DataFrame.
        Input:  DataFrame indexed by date (one row per day), numeric feature columns.
        Output: dict with the analysis results, structure is analysis-specific.
        """


class Analyzer:

    def __init__(self):
        self._analyses:dict[str,Analysis] = {}


    def register(self, name:str, analysis:Analysis):
        self._analyses[name] = analysis


    def run_all(self, df:pd.DataFrame) -> dict[str,dict]:
        return {name: analysis.run(df) for name, analysis in self._analyses.items()}
