"""Utilities for working with tabular (feather) format data."""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_step_data(steps_feather_path: Path) -> pd.DataFrame:
    """Load the JPAL steps.feather file."""
    return pd.read_feather(steps_feather_path)


def create_transition_df(step_df: pd.DataFrame) -> pd.DataFrame:
    """Create a transition-level dataframe from steps.
    
    Each row represents a transition from step_i to step_i+1.
    """
    # Sort by student, question, and step index
    step_df = step_df.sort_values(["student_id", "question_idx", "step_idx"]).copy()
    
    # Get the next step for each row
    step_df["next_step_id"] = step_df.groupby(["student_id", "question_idx"])["step_id"].shift(-1)
    step_df["next_step_idx"] = step_df.groupby(["student_id", "question_idx"])["step_idx"].shift(-1)
    step_df["next_step_text"] = step_df.groupby(["student_id", "question_idx"])["step_text"].shift(-1)
    
    # Filter to only rows that have a next step
    transition_df = step_df[step_df["next_step_id"].notna()].copy()
    
    # Create transition ID
    transition_df["transition_id"] = transition_df.apply(
        lambda row: f"{row['step_id']}->{row['next_step_id']}", 
        axis=1
    )
    
    # Rename columns for clarity
    transition_df = transition_df.rename(columns={
        "step_id": "from_step_id",
        "step_idx": "from_step_idx", 
        "step_text": "from_step_text",
        "next_step_id": "to_step_id",
        "next_step_idx": "to_step_idx",
        "next_step_text": "to_step_text"
    })
    
    # Select relevant columns
    cols = [
        "transition_id", "student_id", "extract_id", "question_id", "question_idx",
        "from_step_id", "from_step_idx", "from_step_text",
        "to_step_id", "to_step_idx", "to_step_text",
        "legible", "ambiguous", "crossed_out"
    ]
    
    return transition_df[cols]


def join_with_questions(transition_df: pd.DataFrame, questions_feather_path: Path) -> pd.DataFrame:
    """Join transition data with question text."""
    question_df = pd.read_feather(questions_feather_path)
    question_cols = ["question_id", "question_text", "instruction", "answer_text"]
    
    return transition_df.merge(
        question_df[question_cols],
        on="question_id",
        how="left"
    )


def join_with_extracts(transition_df: pd.DataFrame, extracts_feather_path: Path) -> pd.DataFrame:
    """Join transition data with extract/student metadata."""
    extract_df = pd.read_feather(extracts_feather_path)
    extract_cols = ["extract_id", "school", "gender", "age", "standard", "domain", "provider"]
    
    return transition_df.merge(
        extract_df[extract_cols],
        on="extract_id",
        how="left"
    )


def prepare_detection_input(
    steps_feather_path: Path,
    questions_feather_path: Path,
    extracts_feather_path: Optional[Path] = None,
    filter_crossed_out: bool = True,
    filter_illegible: bool = True
) -> pd.DataFrame:
    """Prepare a complete transition dataset ready for misconception detection."""
    # Load steps and create transitions
    step_df = load_step_data(steps_feather_path)
    transition_df = create_transition_df(step_df)
    
    # Apply filters
    if filter_crossed_out:
        transition_df = transition_df[~transition_df["crossed_out"]]
    
    if filter_illegible:
        transition_df = transition_df[transition_df["legible"]]
    
    # Join with questions
    transition_df = join_with_questions(transition_df, questions_feather_path)
    
    # Join with extracts if provided
    if extracts_feather_path:
        transition_df = join_with_extracts(transition_df, extracts_feather_path)
    
    return transition_df


def export_for_analysis(
    transition_df: pd.DataFrame,
    output_path: Path,
    format: str = "feather"
):
    """Export transition data with detections for analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "feather":
        transition_df.to_feather(output_path)
    elif format == "csv":
        transition_df.to_csv(output_path, index=False)
    elif format == "parquet":
        transition_df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    print(f"Exported {len(transition_df)} transitions to {output_path}")

