import click

from apple_health_data_parser.parser import HealthDataParser
from apple_health_analytics.preprocessor import Preprocessor
from apple_health_analytics.handlers.steps import StepsHandler
from apple_health_analytics.handlers.heart_rate import HeartRateHandler
from apple_health_analytics.handlers.sleep import SleepHandler
from apple_health_analytics.handlers.hydration import HydrationHandler
from apple_health_analytics.handlers.nutrition import NutritionHandler
from apple_health_analytics.handlers.workout import WorkoutHandler
from apple_health_analytics.analyzer import Analyzer
from apple_health_analytics.analyses.correlation import CorrelationAnalysis
from apple_health_analytics.analyses.weekday import WeekdayPatternAnalysis


def build_preprocessor() -> Preprocessor:
    preprocessor = Preprocessor()
    preprocessor.register("steps", StepsHandler())
    preprocessor.register("heart_rate", HeartRateHandler())
    preprocessor.register("sleep", SleepHandler())
    preprocessor.register("hydration", HydrationHandler())
    preprocessor.register("nutrition", NutritionHandler())
    preprocessor.register("workout", WorkoutHandler())
    return preprocessor


def build_analyzer(max_lag:int, min_abs_corr:float, min_deviation:float) -> Analyzer:
    analyzer = Analyzer()
    analyzer.register("correlation", CorrelationAnalysis(max_lag=max_lag, min_abs_corr=min_abs_corr))
    analyzer.register("weekday", WeekdayPatternAnalysis(min_deviation=min_deviation))
    return analyzer


def print_correlations(result:dict, top_n:int):
    print(f"\n=== Correlations (top {top_n}) ===")
    if not result["pairs"]:
        print("  No correlations above threshold found.")
    for pair in result["pairs"][:top_n]:
        timing = "same day" if pair["lag"] == 0 else f"{pair['lag']} day(s) later"
        print(f"  {pair['cause']} -> {pair['effect']} ({timing}): r={pair['correlation']} [{pair['n_days']} days]")


def print_weekday_patterns(result:dict):
    print("\n=== Weekday patterns ===")
    if not result["notable"]:
        print("  No notable weekday patterns found.")
    for pattern in result["notable"]:
        sign = "+" if pattern["deviation_pct"] > 0 else ""
        print(f"  {pattern['feature']}: {sign}{pattern['deviation_pct']}% on {pattern['weekday']}")


@click.command()
@click.option("--export", "-e", "export_path", required=True, help="path to Apple Health Export.xml")
@click.option("--max-lag", default=1, show_default=True, help="max day offset for lag correlations")
@click.option("--min-corr", default=0.3, show_default=True, help="minimum absolute correlation to report")
@click.option("--min-deviation", default=0.15, show_default=True, help="minimum weekday deviation to report")
@click.option("--top", default=15, show_default=True, help="number of correlation pairs to show")
def main(export_path, max_lag, min_corr, min_deviation, top):
    print(f"Parsing {export_path} ...")
    raw_df = HealthDataParser().parse(export_path)
    print(f"  {len(raw_df)} records")
    daily_df = build_preprocessor().process(raw_df)
    print(f"  {len(daily_df)} days, {len(daily_df.columns)} features")
    results = build_analyzer(max_lag, min_corr, min_deviation).run_all(daily_df)
    print_correlations(results["correlation"], top)
    print_weekday_patterns(results["weekday"])


if __name__ == "__main__":
    main()
