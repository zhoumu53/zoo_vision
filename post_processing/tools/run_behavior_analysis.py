#!/usr/bin/env python3
"""
Behavior Analysis CLI

Command-line interface for running behavior analysis (Stage 4).
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.behavior_analyzer import BehaviorAnalyzer, TrackingDataLoader, TimelineCompressor


def setup_logger(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("behavior_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Behavior analysis: Timeline compression and visualization"
    )
    
    # Required arguments
    parser.add_argument("--input", required=True, help="Input JSONL file (post-processed)")
    parser.add_argument("--output", required=True, help="Output directory")
    
    # Optional parameters
    parser.add_argument("--min-duration", type=float, default=2.0, help="Minimum segment duration (seconds)")
    parser.add_argument("--no-skip-existing", action="store_true", help="Reprocess even if cached results exist")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating visualization plots")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    return parser.parse_args()


def main(args, logger) -> int:
    # Create analyzer
    data_loader = TrackingDataLoader(logger)
    compressor = TimelineCompressor(min_duration=args.min_duration, logger=logger)
    analyzer = BehaviorAnalyzer(data_loader, compressor, logger)
    
    # Run analysis
    df, segments_df = analyzer.analyze(
        jsonl_path=Path(args.input),
        output_dir=Path(args.output),
        skip_if_exists=not args.no_skip_existing,
        save_plots=not args.no_plots,
    )
    
    logger.info("Behavior analysis complete")
    return 0


if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger(args.log_level)
    sys.exit(main(args, logger))