#!/usr/bin/env python3
"""
Language Teaching Script - Teaches the brain to speak through pure exposure.

This script feeds sentences to the brain's LanguageLearner. The brain learns
language patterns statistically - NO hardcoded responses, NO expected outputs.

How it works:
1. Load sentences from a corpus file
2. Feed each sentence to the LanguageLearner
3. The learner tracks word frequencies and discovers patterns
4. Report learning progress

The brain starts mute and learns to speak through exposure.
"""

import sys
import json
import random
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.language.language_learner import LanguageLearner


def load_corpus(corpus_path: Path) -> list[str]:
    """Load sentences from a corpus file (one sentence per line)."""
    sentences = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip comments and empty lines
                sentences.append(line)
    return sentences


def teach(learner: LanguageLearner, sentences: list[str], iterations: int = 3,
          shuffle: bool = True, verbose: bool = True) -> dict:
    """
    Teach the brain by pure exposure.

    No hardcoded responses. No expected outputs.
    Just feed sentences and let the brain learn.

    Args:
        learner: The LanguageLearner instance
        sentences: List of sentences to learn from
        iterations: Number of times to repeat the corpus
        shuffle: Whether to shuffle sentences between iterations
        verbose: Whether to print progress

    Returns:
        Statistics about what was learned
    """
    total_patterns_discovered = 0
    total_words_observed = 0
    patterns_per_iteration = []

    for iteration in range(iterations):
        if shuffle:
            random.shuffle(sentences)

        iteration_patterns = 0

        for i, sentence in enumerate(sentences):
            result = learner.learn_from_sentence(sentence)

            if result['learned']:
                total_words_observed += result.get('words_observed', 0)
                new_patterns = result.get('patterns_discovered', [])
                iteration_patterns += len(new_patterns)
                total_patterns_discovered += len(new_patterns)

                if verbose and new_patterns:
                    print(f"  Discovered: {new_patterns}")

        patterns_per_iteration.append(iteration_patterns)

        if verbose:
            stats = learner.get_stats()
            print(f"\nIteration {iteration + 1}/{iterations}:")
            print(f"  Vocabulary: {stats['vocabulary_size']} words")
            print(f"  Function words learned: {stats['learned_function_words']}")
            print(f"  Content words learned: {stats['learned_content_words']}")
            print(f"  Patterns discovered this iteration: {iteration_patterns}")
            print(f"  Total patterns: {stats['total_patterns']}")

    return {
        'total_patterns_discovered': total_patterns_discovered,
        'total_words_observed': total_words_observed,
        'patterns_per_iteration': patterns_per_iteration,
        'final_stats': learner.get_stats(),
    }


def test_generation(learner: LanguageLearner, verbose: bool = True) -> list[dict]:
    """
    Test the brain's ability to generate responses.
    This is observation only - we don't assert specific outputs.
    """
    test_cases = [
        # Basic facts
        [("dog", "is", "animal")],
        [("cat", "is", "pet")],
        [("bird", "can", "fly")],
        # Self-reference
        [("xemsa", "is", "intelligence")],
        [("I", "am", "learning")],
        # Unknown (should return empty if not enough patterns)
        [("quantum", "is", "physics")],
    ]

    results = []

    if verbose:
        print("\n--- Generation Test ---")
        print("Testing if the brain can express knowledge using learned patterns:\n")

    for facts in test_cases:
        response = learner.generate_response(facts)
        result = {
            'facts': facts,
            'response': response,
            'can_express': bool(response),
        }
        results.append(result)

        if verbose:
            fact_str = ", ".join([f"{s} {r} {o}" for s, r, o in facts])
            print(f"  Knowledge: {fact_str}")
            print(f"  Output: '{response}' {'(silence)' if not response else ''}")
            print()

    return results


def main():
    parser = argparse.ArgumentParser(description='Teach the brain to speak through exposure')
    parser.add_argument('--corpus', type=str, default='data/teaching_corpus.txt',
                        help='Path to corpus file (one sentence per line)')
    parser.add_argument('--iterations', type=int, default=3,
                        help='Number of times to repeat the corpus')
    parser.add_argument('--output', type=str, default='data/language_patterns.json',
                        help='Path to save learned patterns')
    parser.add_argument('--load', type=str, default=None,
                        help='Path to load existing patterns from')
    parser.add_argument('--no-shuffle', action='store_true',
                        help='Do not shuffle sentences between iterations')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    parser.add_argument('--test', action='store_true',
                        help='Run generation tests after teaching')

    args = parser.parse_args()
    verbose = not args.quiet

    # Create or load learner
    output_path = Path(args.output)
    learner = LanguageLearner(storage_path=output_path if args.load else None)

    if args.load:
        load_path = Path(args.load)
        if load_path.exists():
            learner.load(load_path)
            if verbose:
                print(f"Loaded existing patterns from {load_path}")
                stats = learner.get_stats()
                print(f"  Existing vocabulary: {stats['vocabulary_size']} words")
                print(f"  Existing patterns: {stats['total_patterns']}")
                print()

    # Load corpus
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"Error: Corpus file not found: {corpus_path}")
        print("Please create a corpus file with one sentence per line.")
        print("Example content:")
        print("  The dog is an animal.")
        print("  A cat is a pet.")
        print("  Birds can fly.")
        sys.exit(1)

    sentences = load_corpus(corpus_path)

    if verbose:
        print(f"Teaching from {corpus_path}")
        print(f"Corpus contains {len(sentences)} sentences")
        print(f"Running {args.iterations} iterations")
        print()

    # Teach
    results = teach(
        learner=learner,
        sentences=sentences,
        iterations=args.iterations,
        shuffle=not args.no_shuffle,
        verbose=verbose,
    )

    # Save learned patterns
    output_path.parent.mkdir(parents=True, exist_ok=True)
    learner.save(output_path)

    if verbose:
        print(f"\n--- Final Statistics ---")
        stats = results['final_stats']
        print(f"Vocabulary: {stats['vocabulary_size']} words")
        print(f"Function words: {stats['learned_function_words']}")
        print(f"Content words: {stats['learned_content_words']}")
        print(f"Patterns discovered: {stats['total_patterns']}")
        print(f"\nTop function words: {stats['function_word_examples'][:10]}")
        print(f"\nTop patterns:")
        for template, freq in stats['patterns_by_frequency'][:5]:
            print(f"  {template} (used {freq}x)")
        print(f"\nPatterns saved to: {output_path}")

    # Optionally run generation tests
    if args.test:
        test_generation(learner, verbose=verbose)

    return results


if __name__ == "__main__":
    main()
