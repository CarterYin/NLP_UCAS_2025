"""Command-line tool to verify Zipf's law on English word frequencies."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

from compute_cd2 import analyze_words


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a Zipf regression on English word frequencies.",
    )
    parser.add_argument(
        "corpus",
        type=Path,
        help="Path to the UTF-8 encoded corpus file to analyze.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=200,
        help="Number of top-ranked words to include in the regression (default: 200).",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Treat uppercase and lowercase words as distinct tokens.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=10,
        help="Show the first K rows of the rank-frequency summary (default: 10).",
    )
    return parser.parse_args()


def load_frequencies(path: Path, case_sensitive: bool, top_n: int) -> tuple[list[tuple[int, float]], int, int]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    counts, total, _ = analyze_words(text, case_sensitive)
    pairs = [
        (rank + 1, count / total)
        for rank, (_, count) in enumerate(counts.most_common(top_n))
    ]
    return pairs, total, len(counts)


def linear_regression(pairs: Iterable[tuple[int, float]]) -> tuple[float, float, float]:
    log_ranks = [math.log10(rank) for rank, _ in pairs]
    log_freqs = [math.log10(freq) for _, freq in pairs]
    n = len(log_ranks)
    mean_x = sum(log_ranks) / n
    mean_y = sum(log_freqs) / n
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_ranks, log_freqs))
    denominator = sum((x - mean_x) ** 2 for x in log_ranks)
    slope = numerator / denominator
    intercept = mean_y - slope * mean_x
    mse = sum(
        (y - (slope * x + intercept)) ** 2
        for x, y in zip(log_ranks, log_freqs)
    ) / n
    return slope, intercept, mse


def main() -> None:
    args = parse_args()
    pairs, total, unique = load_frequencies(args.corpus, args.case_sensitive, args.top)
    if not pairs:
        raise ValueError("no words were found in the corpus")

    slope, intercept, mse = linear_regression(pairs)

    print(f"Corpus: {args.corpus}")
    print(f"Total words: {total}")
    print(f"Unique words: {unique}")
    print(f"Top ranks used: 1-{len(pairs)}")
    print(
        "Linear fit (log10 rank vs log10 freq): "
        f"slope={slope:.3f}, intercept={intercept:.3f}, mse={mse:.5f}"
    )
    print()
    print("rank freq rank*freq log10(rank) log10(freq)")
    sample_pairs = pairs[: max(args.show, 0)]
    for rank, freq in sample_pairs:
        log_rank = math.log10(rank)
        log_freq = math.log10(freq)
        print(
            f"{rank:>4}  {freq:>0.6f}  {rank * freq:>0.6f}  "
            f"{log_rank:>10.6f}  {log_freq:>10.6f}"
        )


if __name__ == "__main__":
    main()
