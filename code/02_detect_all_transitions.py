#!/usr/bin/env python3
"""Detect misconceptions on all transitions from feather file.

MAKES API CALLS to OpenAI (GPT-5) and/or Gemini.

This script reads transitions with correctness already computed,
runs LLM detection, and exports results.
"""

import argparse
import pathlib
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from detector import MisconceptionDetector
from schemas import TransitionResult


def detect_batch(
    transitions: pd.DataFrame,
    detector: MisconceptionDetector,
    model: str = "both",
    max_workers: int = 4
):
    """Run detection on a batch of transitions."""
    results = []
    
    def process_row(idx, row):
        try:
            result = detector.detect_transition(
                step_from=row.get("step_text", ""),
                step_to=row.get("step_text_next", ""),
                question=row.get("question_text"),
                question_id=row.get("question_id"),
                transition_index=idx,
                model=model,
                domain="linear_equations"
            )
            
            # Convert to dict and add metadata
            result_dict = result.model_dump()
            result_dict["transition_id"] = row.get("transition_id")
            result_dict["student_id"] = row.get("student_id")
            result_dict["implied"] = row.get("implied")  # Mathematical correctness
            
            return idx, result_dict
        except Exception as e:
            print(f"  Error on transition {idx}: {e}")
            return idx, None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_row, idx, row): idx
            for idx, row in transitions.iterrows()
        }
        
        for future in as_completed(futures):
            try:
                idx, result_dict = future.result()
                if result_dict:
                    results.append(result_dict)
                    if len(results) % 50 == 0:
                        print(f"  Processed {len(results)}...")
            except Exception as e:
                print(f"  Exception: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Detect misconceptions on all transitions"
    )
    parser.add_argument(
        "--transitions-file",
        type=pathlib.Path,
        required=True,
        help="Path to transitions.feather file"
    )
    parser.add_argument(
        "--output-file",
        type=pathlib.Path,
        required=True,
        help="Output file (.feather or .csv)"
    )
    parser.add_argument(
        "--model",
        choices=["gemini", "openai", "both"],
        default="both",
        help="Which LLM to use"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of transitions (for testing)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    
    args = parser.parse_args()
    
    # Load transitions
    print(f"Loading transitions from {args.transitions_file}...")
    transitions_df = pd.read_feather(args.transitions_file)
    print(f"  Loaded {len(transitions_df)} transitions")
    
    if args.limit:
        transitions_df = transitions_df.head(args.limit)
        print(f"  Limited to {len(transitions_df)} for testing")
    
    # Initialize detector
    print(f"\nInitializing detector with model: {args.model}")
    try:
        use_gemini = args.model in ["gemini", "both"]
        use_openai = args.model in ["openai", "both"]
        detector = MisconceptionDetector(use_gemini=use_gemini, use_openai=use_openai)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Run detection
    print(f"\nDetecting misconceptions...")
    print("=" * 70)
    results = detect_batch(
        transitions_df,
        detector,
        model=args.model,
        max_workers=args.max_workers
    )
    print("=" * 70)
    print(f"Completed {len(results)} transitions")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Flatten nested dicts
    if "openai" in results_df.columns:
        openai_df = pd.json_normalize(results_df["openai"])
        openai_df.columns = [f"openai_{col}" for col in openai_df.columns]
        results_df = pd.concat([results_df.drop(columns=["openai"]), openai_df], axis=1)
    
    if "gemini" in results_df.columns:
        gemini_df = pd.json_normalize(results_df["gemini"])
        gemini_df.columns = [f"gemini_{col}" for col in gemini_df.columns]
        results_df = pd.concat([results_df.drop(columns=["gemini"]), gemini_df], axis=1)
    
    # Save
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    if args.output_file.suffix == ".csv":
        results_df.to_csv(args.output_file, index=False)
    else:
        results_df.to_feather(args.output_file)
    
    print(f"\n✓ Results saved to: {args.output_file}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total: {len(results_df)}")
    if "openai_misconception_a" in results_df.columns:
        print(f"  OpenAI A: {results_df['openai_misconception_a'].sum()}")
        print(f"  OpenAI B: {results_df['openai_misconception_b'].sum()}")
        print(f"  OpenAI C: {results_df['openai_misconception_c'].sum()}")
    if "gemini_misconception_a" in results_df.columns:
        print(f"  Gemini A: {results_df['gemini_misconception_a'].sum()}")
        print(f"  Gemini B: {results_df['gemini_misconception_b'].sum()}")
        print(f"  Gemini C: {results_df['gemini_misconception_c'].sum()}")


if __name__ == "__main__":
    main()

