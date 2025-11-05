#!/usr/bin/env python3
"""Filter transitions to only those with human labels (misconceptions_apo_v1.csv).

Matching logic:
1. Match school code from human labels to students (extract from school_id)
2. Match problem_set_id (case-insensitive)
3. Match student_index to the ordering within school

NO API CALLS - just data filtering.
"""

import argparse
import pathlib
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Filter data to only students/questions with human labels"
    )
    parser.add_argument(
        "--labels-csv",
        type=pathlib.Path,
        default=pathlib.Path("data/misconceptions_apo_v1.csv"),
        help="Path to misconceptions_apo_v1.csv (human labels)"
    )
    parser.add_argument(
        "--students-file",
        type=pathlib.Path,
        default=pathlib.Path("data/tabular/convergent/linear_equations/students.csv"),
        help="Path to students file (CSV or feather)"
    )
    parser.add_argument(
        "--transitions-file",
        type=pathlib.Path,
        default=pathlib.Path("data/tabular/convergent/linear_equations/transitions.csv"),
        help="Path to transitions file (CSV or feather, with 'implied' column)"
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("data/linear_equations"),
        help="Output directory for filtered data"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("FILTERING DATA TO HUMAN-LABELED STUDENTS/QUESTIONS")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    labels_df = pd.read_csv(args.labels_csv)
    
    # Read students (auto-detect CSV or feather)
    if args.students_file.suffix == '.csv':
        students_df = pd.read_csv(args.students_file)
    else:
        students_df = pd.read_feather(args.students_file)
    
    # Read transitions (auto-detect CSV or feather)
    if args.transitions_file.suffix == '.csv':
        transitions_df = pd.read_csv(args.transitions_file)
    else:
        transitions_df = pd.read_feather(args.transitions_file)
    
    print(f"  Human labels: {len(labels_df)}")
    print(f"  Students: {len(students_df)}")
    print(f"  Transitions: {len(transitions_df)}")
    print(f"  Labeled schools: {sorted(labels_df['school'].unique())}")
    
    # Match by school (contained in school_id) + json_filename = student_index
    print("\nMatching human labels to students by school + json_filename...")
    labels_matched = []
    
    for _, label_row in labels_df.iterrows():
        school = label_row['school']
        student_idx = label_row['student_index']
        
        # Find student where school is in school_id AND json_filename matches student_index
        student_match = students_df[
            (students_df['school_id'].str.contains(school, na=False, case=False)) &
            (students_df['json_filename'] == student_idx)
        ]
        
        if len(student_match) == 0:
            print(f"  ⚠ No student found for school={school}, json_filename={student_idx}")
            continue
        
        if len(student_match) > 1:
            print(f"  ⚠ Multiple students ({len(student_match)}) found for school={school}, json_filename={student_idx}")
            continue
        
        student_row = student_match.iloc[0]
        
        labels_matched.append({
            'student_id': student_row['student_id'],
            'question_id': label_row['question_id'],
            'school': label_row['school'],
            'problem_set_id': label_row['problem_set_id'],
            'student_index': label_row['student_index'],
            'apo': label_row['apo']
        })
    
    labels_matched_df = pd.DataFrame(labels_matched)
    print(f"  Matched {len(labels_matched_df)} labeled records to students")
    print(f"  Unique students: {labels_matched_df['student_id'].nunique()}")
    
    if len(labels_matched_df) == 0:
        print("\n⚠ WARNING: No matches found! Check school codes and problem_set_ids")
        return
    
    # Filter students
    print("\nFiltering students...")
    matched_student_ids = set(labels_matched_df['student_id'].unique())
    filtered_students = students_df[students_df['student_id'].isin(matched_student_ids)].copy()
    print(f"  Kept {len(filtered_students)} students")
    
    # Build set of (student_id, question_id) pairs for transition filtering
    # Question IDs in transitions are like "convergent1-x000-q00"
    # Human label question_ids are integers like 0, 1, 2
    print("\nBuilding question filter...")
    student_question_pairs = set()
    
    for _, row in labels_matched_df.iterrows():
        student_id = row['student_id']
        question_num = row['question_id']
        # Match any question_id that starts with student_id and has this question number
        # Format: {student_id}-q{question_num:02d}
        question_id_pattern = f"{student_id}-q{question_num:02d}"
        student_question_pairs.add((student_id, question_id_pattern))
    
    print(f"  Built {len(student_question_pairs)} (student, question) filters")
    
    # Filter transitions
    print("\nFiltering transitions...")
    
    def matches_filter(row):
        student_id = row['student_id']
        question_id = row['question_id']
        # Check if this (student_id, question_id) matches any pattern
        for (filter_student, filter_question) in student_question_pairs:
            if student_id == filter_student and question_id.startswith(filter_question):
                return True
        return False
    
    filtered_transitions = transitions_df[
        transitions_df.apply(matches_filter, axis=1)
    ].copy()
    
    print(f"  Original transitions: {len(transitions_df)}")
    print(f"  Filtered transitions: {len(filtered_transitions)}")
    
    # Save filtered data
    print("\nSaving filtered data...")
    
    # Save students
    students_output = args.output_dir / "students.feather"
    filtered_students.to_feather(students_output)
    print(f"  ✓ Students: {students_output}")
    
    # Save transitions
    transitions_output = args.output_dir / "transitions.feather"
    filtered_transitions.to_feather(transitions_output)
    print(f"  ✓ Transitions: {transitions_output}")
    
    # Also save the matched human labels for reference
    labels_output = args.output_dir / "human_labels_matched.csv"
    labels_matched_df.to_csv(labels_output, index=False)
    print(f"  ✓ Human labels matched: {labels_output}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Schools with human labels: {labels_df['school'].nunique()} ({', '.join(sorted(labels_df['school'].unique()))})")
    print(f"Students filtered: {len(filtered_students)}")
    print(f"Transitions filtered: {len(filtered_transitions)}")
    print(f"Transitions per student (avg): {len(filtered_transitions) / len(filtered_students):.1f}")
    print(f"\nOutput directory: {args.output_dir}")


if __name__ == "__main__":
    main()

