"""Compute single-character probabilities and entropy for English text corpora."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Iterable


def iter_english_letters(text: str, case_sensitive: bool) -> Iterable[str]:
	"""Yield ASCII alphabetic characters, normalizing case when requested."""
	for char in text:
		if char.isascii() and char.isalpha():
			yield char if case_sensitive else char.lower()


def analyze_characters(text: str, case_sensitive: bool) -> tuple[Counter[str], int, float]:
	counts = Counter(iter_english_letters(text, case_sensitive))
	total = sum(counts.values())
	if not total:
		raise ValueError("input corpus does not contain any ASCII alphabetic characters")
	probabilities = (count / total for count in counts.values())
	entropy = -sum(p * math.log2(p) for p in probabilities)
	return counts, total, entropy


def format_top_characters(counts: Counter[str], total: int, top_k: int) -> str:
	lines = []
	header = "rank char count prob information(bits) contribution"
	lines.append(header)
	for rank, (char, count) in enumerate(counts.most_common(top_k), 1):
		probability = count / total
		information_bits = -math.log2(probability)
		contribution = probability * information_bits
		lines.append(
			f"{rank:>4}  {char}   {count:>8}  {probability:>0.6f}"
			f"      {information_bits:>10.6f}       {contribution:>10.6f}"
		)
	return "\n".join(lines)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Compute English letter probabilities and Shannon entropy."
	)
	parser.add_argument(
		"corpus",
		type=Path,
		help="Path to the UTF-8 encoded corpus file to analyze.",
	)
	parser.add_argument(
		"--top",
		type=int,
		default=20,
		help="Show the top-K most frequent letters (default: 20).",
	)
	parser.add_argument(
		"--case-sensitive",
		action="store_true",
		help="Treat uppercase and lowercase letters as distinct symbols.",
	)
	args = parser.parse_args()

	text = args.corpus.read_text(encoding="utf-8", errors="ignore")
	counts, total, entropy = analyze_characters(text, args.case_sensitive)
	unique = len(counts)

	print(f"Corpus: {args.corpus}")
	print(f"Total letters: {total}")
	print(f"Unique letters: {unique}")
	print(f"Shannon entropy: {entropy:.6f} bits")

	if args.top > 0:
		print("\nTop letters:")
		print(format_top_characters(counts, total, args.top))


if __name__ == "__main__":
	main()
