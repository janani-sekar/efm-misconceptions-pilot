"""Pydantic schemas for structured LLM outputs."""

from typing import Optional
from pydantic import BaseModel, Field


class TransitionDetection(BaseModel):
    """Detection result for a single step transition."""
    
    misconception_a: bool = Field(
        description="Moving terms without changing signs"
    )
    explanation_a: str = Field(
        description="Explanation for why Misconception A was or was not detected"
    )
    misconception_b: bool = Field(
        description="Distributive property error"
    )
    explanation_b: str = Field(
        description="Explanation for why Misconception B was or was not detected"
    )
    misconception_c: bool = Field(
        description="Combining unlike terms"
    )
    explanation_c: str = Field(
        description="Explanation for why Misconception C was or was not detected"
    )


class TransitionResult(BaseModel):
    """Complete result for a transition including metadata."""
    
    step_from: str
    step_to: str
    question: Optional[str] = None
    question_id: Optional[int] = None
    transition_index: int
    
    # Detection results from each model
    gemini: Optional[TransitionDetection] = None
    openai: Optional[TransitionDetection] = None
    
    # Error tracking
    gemini_error: Optional[str] = None
    openai_error: Optional[str] = None


class QuestionSummary(BaseModel):
    """Aggregated detections for a question."""
    
    question_id: int
    question_text: str
    total_transitions: int
    
    # Counts per misconception
    misconception_a_count: int = 0
    misconception_b_count: int = 0
    misconception_c_count: int = 0
    
    # Agreement between models
    model_agreement_rate: Optional[float] = None


class StudentSummary(BaseModel):
    """Aggregated detections for a student."""
    
    problem_set_id: str
    gender: Optional[str] = None
    age: Optional[int] = None
    standard: Optional[int] = None
    
    total_questions: int
    total_transitions: int
    
    # Misconception counts across all transitions
    misconception_a_total: int = 0
    misconception_b_total: int = 0
    misconception_c_total: int = 0
    
    # Per-question summaries
    questions: list[QuestionSummary] = []

