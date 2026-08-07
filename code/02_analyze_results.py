#!/usr/bin/env python3
"""Analyze LLM detection results with aggregations and visualizations.

NO API CALLS - just analysis and plotting.
"""

import argparse
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List


def flatten_detections(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten nested detection results into columns."""
    rows = []
    
    for _, row in df.iterrows():
        base = {
            'transition_id': row.get('transition_id'),
            'student_id': row.get('student_id'),
            'question_id': row.get('question_id'),
            'step_from': row.get('step_from'),
            'step_to': row.get('step_to'),
            'implied': row.get('implied'),
        }
        
        # Extract Gemini results
        if 'gemini' in row and pd.notna(row['gemini']):
            gemini = row['gemini']
            if isinstance(gemini, dict):
                base['gemini_misconception_a'] = gemini.get('misconception_a')
                base['gemini_misconception_b'] = gemini.get('misconception_b')
                base['gemini_misconception_c'] = gemini.get('misconception_c')
                base['gemini_explanation_a'] = gemini.get('explanation_a')
                base['gemini_explanation_b'] = gemini.get('explanation_b')
                base['gemini_explanation_c'] = gemini.get('explanation_c')
        
        # Extract OpenAI results
        if 'openai' in row and pd.notna(row['openai']):
            openai = row['openai']
            if isinstance(openai, dict):
                base['openai_misconception_a'] = openai.get('misconception_a')
                base['openai_misconception_b'] = openai.get('misconception_b')
                base['openai_misconception_c'] = openai.get('misconception_c')
                base['openai_explanation_a'] = openai.get('explanation_a')
                base['openai_explanation_b'] = openai.get('explanation_b')
                base['openai_explanation_c'] = openai.get('explanation_c')
        
        rows.append(base)
    
    return pd.DataFrame(rows)


def aggregate_to_question(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transition-level results to question level."""
    agg_dict = {}
    
    # Count misconceptions per question
    for model in ['gemini', 'openai']:
        for misc in ['a', 'b', 'c']:
            col = f'{model}_misconception_{misc}'
            if col in df.columns:
                agg_dict[f'{model}_{misc}_count'] = (col, 'sum')
                agg_dict[f'{model}_{misc}_rate'] = (col, 'mean')
    
    # Add transition count
    agg_dict['n_transitions'] = ('transition_id', 'count')
    
    # Add correctness if available
    if 'implied' in df.columns:
        agg_dict['correct_count'] = ('implied', lambda x: (x==1).sum())
        agg_dict['incorrect_count'] = ('implied', lambda x: (x==0).sum())
        agg_dict['correctness_rate'] = ('implied', 'mean')
    
    question_df = df.groupby(['question_id', 'student_id']).agg(**agg_dict).reset_index()
    return question_df


def aggregate_to_student(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transition-level results to student level."""
    agg_dict = {}
    
    # Count misconceptions per student
    for model in ['gemini', 'openai']:
        for misc in ['a', 'b', 'c']:
            col = f'{model}_misconception_{misc}'
            if col in df.columns:
                agg_dict[f'{model}_{misc}_total'] = (col, 'sum')
                agg_dict[f'{model}_{misc}_rate'] = (col, 'mean')
    
    # Add counts
    agg_dict['n_transitions'] = ('transition_id', 'count')
    agg_dict['n_questions'] = ('question_id', 'nunique')
    
    # Add correctness if available
    if 'implied' in df.columns:
        agg_dict['correct_count'] = ('implied', lambda x: (x==1).sum())
        agg_dict['incorrect_count'] = ('implied', lambda x: (x==0).sum())
        agg_dict['correctness_rate'] = ('implied', 'mean')
    
    student_df = df.groupby('student_id').agg(**agg_dict).reset_index()
    return student_df


def create_visualizations(df: pd.DataFrame, output_dir: pathlib.Path):
    """Create sexy visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150
    
    # 1. Misconception Detection Rates by Model
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rates = []
    for model in ['gemini', 'openai']:
        for misc in ['a', 'b', 'c']:
            col = f'{model}_misconception_{misc}'
            if col in df.columns:
                rate = df[col].mean() * 100
                rates.append({
                    'Model': model.upper(),
                    'Misconception': f'Type {misc.upper()}',
                    'Detection Rate (%)': rate
                })
    
    if rates:
        rates_df = pd.DataFrame(rates)
        sns.barplot(data=rates_df, x='Misconception', y='Detection Rate (%)', 
                   hue='Model', ax=ax, palette='Set2')
        ax.set_title('Misconception Detection Rates by Model', fontsize=14, fontweight='bold')
        ax.set_ylabel('Detection Rate (%)', fontsize=12)
        ax.set_xlabel('Misconception Type', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / 'detection_rates.png', bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_dir / 'detection_rates.png'}")
    
    # 2. Model Agreement Heatmap
    if 'gemini_misconception_a' in df.columns and 'openai_misconception_a' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        agreement_matrix = []
        for misc in ['a', 'b', 'c']:
            gemini_col = f'gemini_misconception_{misc}'
            openai_col = f'openai_misconception_{misc}'
            
            agreement = (df[gemini_col] == df[openai_col]).mean() * 100
            gemini_rate = df[gemini_col].mean() * 100
            openai_rate = df[openai_col].mean() * 100
            
            agreement_matrix.append([gemini_rate, openai_rate, agreement])
        
        agreement_df = pd.DataFrame(
            agreement_matrix,
            index=['Type A', 'Type B', 'Type C'],
            columns=['Gemini Rate', 'OpenAI Rate', 'Agreement']
        )
        
        sns.heatmap(agreement_df, annot=True, fmt='.1f', cmap='YlGnBu', 
                   ax=ax, cbar_kws={'label': 'Percentage (%)'})
        ax.set_title('Model Comparison: Detection Rates & Agreement', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'model_agreement.png', bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_dir / 'model_agreement.png'}")
    
    # 3. Student-Level Analysis
    if 'student_id' in df.columns:
        student_df = aggregate_to_student(df)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 3a. Distribution of transitions per student
        axes[0, 0].hist(student_df['n_transitions'], bins=20, 
                       color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Transitions per Student', fontweight='bold')
        axes[0, 0].set_xlabel('Number of Transitions')
        axes[0, 0].set_ylabel('Number of Students')
        
        # 3b. Distribution of questions per student
        axes[0, 1].hist(student_df['n_questions'], bins=20, 
                       color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('Questions per Student', fontweight='bold')
        axes[0, 1].set_xlabel('Number of Questions')
        axes[0, 1].set_ylabel('Number of Students')
        
        # 3c. Misconception rate distribution (Gemini)
        if 'gemini_a_rate' in student_df.columns:
            student_df['gemini_any_misc_rate'] = student_df[[
                'gemini_a_rate', 'gemini_b_rate', 'gemini_c_rate'
            ]].max(axis=1) * 100
            
            axes[1, 0].hist(student_df['gemini_any_misc_rate'], bins=20, 
                           color='mediumseagreen', edgecolor='black')
            axes[1, 0].set_title('Student Misconception Rates (Gemini)', fontweight='bold')
            axes[1, 0].set_xlabel('Max Misconception Rate (%)')
            axes[1, 0].set_ylabel('Number of Students')
        
        # 3d. Correctness vs Misconceptions
        if 'correctness_rate' in student_df.columns and 'gemini_any_misc_rate' in student_df.columns:
            axes[1, 1].scatter(student_df['correctness_rate'] * 100, 
                             student_df['gemini_any_misc_rate'],
                             alpha=0.6, color='purple')
            axes[1, 1].set_title('Correctness vs Misconception Rate', fontweight='bold')
            axes[1, 1].set_xlabel('Mathematical Correctness (%)')
            axes[1, 1].set_ylabel('Misconception Rate (%)')
            
            # Add trend line
            if len(student_df) > 1:
                z = np.polyfit(student_df['correctness_rate'] * 100, 
                              student_df['gemini_any_misc_rate'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(student_df['correctness_rate'].min() * 100,
                                    student_df['correctness_rate'].max() * 100, 100)
                axes[1, 1].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'student_analysis.png', bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_dir / 'student_analysis.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze misconception detection results with aggregations and visualizations"
    )
    parser.add_argument(
        "results_file",
        type=pathlib.Path,
        help="Results file (.feather or .csv)"
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results"),
        help="Directory to save outputs (default: results/)"
    )
    parser.add_argument(
        "--save-aggregations",
        action="store_true",
        help="Save aggregated data to CSV"
    )
    
    args = parser.parse_args()
    
    if not args.results_file.exists():
        print(f"❌ Error: File not found: {args.results_file}")
        return
    
    # Load results
    print(f"📊 Loading results from {args.results_file}...")
    if args.results_file.suffix == ".csv":
        df = pd.read_csv(args.results_file)
    else:
        df = pd.read_feather(args.results_file)
    
    print(f"   Loaded {len(df)} transitions")
    
    # Flatten if needed
    if 'gemini' in df.columns and isinstance(df['gemini'].iloc[0], dict):
        print("   Flattening nested detection results...")
        df = flatten_detections(df)
    
    # Print summary
    print("\n" + "="*70)
    print("TRANSITION-LEVEL SUMMARY")
    print("="*70)
    
    for model in ["gemini", "openai"]:
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
    
    # Aggregate to question level
    if 'question_id' in df.columns:
        print("\n" + "="*70)
        print("QUESTION-LEVEL AGGREGATION")
        print("="*70)
        question_df = aggregate_to_question(df)
        print(f"  {len(question_df)} unique questions")
        print(f"  Avg transitions per question: {question_df['n_transitions'].mean():.1f}")
        
        if args.save_aggregations:
            out_file = args.output_dir / "question_aggregates.csv"
            args.output_dir.mkdir(parents=True, exist_ok=True)
            question_df.to_csv(out_file, index=False)
            print(f"  ✓ Saved: {out_file}")
    
    # Aggregate to student level
    if 'student_id' in df.columns:
        print("\n" + "="*70)
        print("STUDENT-LEVEL AGGREGATION")
        print("="*70)
        student_df = aggregate_to_student(df)
        print(f"  {len(student_df)} unique students")
        print(f"  Avg transitions per student: {student_df['n_transitions'].mean():.1f}")
        print(f"  Avg questions per student: {student_df['n_questions'].mean():.1f}")
        
        if args.save_aggregations:
            out_file = args.output_dir / "student_aggregates.csv"
            student_df.to_csv(out_file, index=False)
            print(f"  ✓ Saved: {out_file}")
    
    # Create visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    create_visualizations(df, args.output_dir)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()

