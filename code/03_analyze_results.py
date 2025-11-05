#!/usr/bin/env python3
"""Analyze LLM detection results.

NO API CALLS - just analysis.
"""

import argparse
import pathlib
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Analyze misconception detection results"
    )
    parser.add_argument(
        "results_file",
        type=pathlib.Path,
        help="Results file (.feather or .csv)"
    )
    
    args = parser.parse_args()
    
    if not args.results_file.exists():
        print(f"Error: File not found: {args.results_file}")
        return
    
    # Load results
    print(f"Loading results from {args.results_file}...")
    if args.results_file.suffix == ".csv":
        df = pd.read_csv(args.results_file)
    else:
        df = pd.read_feather(args.results_file)
    
    print(f"Loaded {len(df)} transitions")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for model in ["openai", "gemini"]:
        col_a = f"{model}_misconception_a"
        if col_a in df.columns:
            print(f"\n{model.upper()}:")
            print(f"  Misconception A: {df[col_a].sum()} ({df[col_a].mean()*100:.1f}%)")
            print(f"  Misconception B: {df[f'{model}_misconception_b'].sum()} ({df[f'{model}_misconception_b'].mean()*100:.1f}%)")
            print(f"  Misconception C: {df[f'{model}_misconception_c'].sum()} ({df[f'{model}_misconception_c'].mean()*100:.1f}%)")
    
    # Model agreement
    if "openai_misconception_a" in df.columns and "gemini_misconception_a" in df.columns:
        print("\nMODEL AGREEMENT:")
        for misc in ["a", "b", "c"]:
            openai_col = f"openai_misconception_{misc}"
            gemini_col = f"gemini_misconception_{misc}"
            agreement = (df[openai_col] == df[gemini_col]).mean()
            print(f"  Misconception {misc.upper()}: {agreement*100:.1f}%")
    
    # Mathematical correctness
    if "implied" in df.columns:
        print("\nMATHEMATICAL CORRECTNESS:")
        print(f"  Correct: {(df['implied']==1).sum()} ({(df['implied']==1).mean()*100:.1f}%)")
        print(f"  Incorrect: {(df['implied']==0).sum()} ({(df['implied']==0).mean()*100:.1f}%)")
        print(f"  Unknown: {df['implied'].isna().sum()}")


if __name__ == "__main__":
    main()

