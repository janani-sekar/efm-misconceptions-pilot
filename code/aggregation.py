"""Aggregation utilities for transitioning from transition-level to question/student-level summaries."""

from typing import List, Dict, Any, Optional
from collections import defaultdict
from schemas import TransitionResult, QuestionSummary, StudentSummary, TransitionDetection


def aggregate_to_question(
    transitions: List[TransitionResult],
    model: str = "gemini"
) -> QuestionSummary:
    """Aggregate transition-level detections to question level.
    
    Args:
        transitions: List of TransitionResult for a single question
        model: Which model's results to aggregate ("gemini" or "openai")
        
    Returns:
        QuestionSummary with aggregated counts
    """
    if not transitions:
        raise ValueError("No transitions provided")
    
    # Get question metadata from first transition
    first = transitions[0]
    question_id = first.question_id if first.question_id is not None else 0
    question_text = first.question or ""
    
    # Count misconceptions
    a_count = 0
    b_count = 0
    c_count = 0
    total_valid = 0
    agreements = 0
    
    for trans in transitions:
        # Get the detection result for specified model
        detection: Optional[TransitionDetection] = None
        if model == "gemini":
            detection = trans.gemini
        elif model == "openai":
            detection = trans.openai
        else:
            raise ValueError(f"Unknown model: {model}")
        
        if detection is None:
            continue
        
        total_valid += 1
        
        # Count each misconception
        if detection.misconception_a:
            a_count += 1
        if detection.misconception_b:
            b_count += 1
        if detection.misconception_c:
            c_count += 1
        
        # Check agreement between models if both present
        if trans.gemini and trans.openai:
            if (trans.gemini.misconception_a == trans.openai.misconception_a and
                trans.gemini.misconception_b == trans.openai.misconception_b and
                trans.gemini.misconception_c == trans.openai.misconception_c):
                agreements += 1
    
    # Calculate agreement rate
    agreement_rate = None
    transitions_with_both = sum(1 for t in transitions if t.gemini and t.openai)
    if transitions_with_both > 0:
        agreement_rate = agreements / transitions_with_both
    
    return QuestionSummary(
        question_id=question_id,
        question_text=question_text,
        total_transitions=len(transitions),
        misconception_a_count=a_count,
        misconception_b_count=b_count,
        misconception_c_count=c_count,
        model_agreement_rate=agreement_rate
    )


def aggregate_to_student(
    all_transitions: List[TransitionResult],
    student_metadata: Dict[str, Any],
    model: str = "gemini"
) -> StudentSummary:
    """Aggregate transition-level detections to student level.
    
    Args:
        all_transitions: All TransitionResult for a student across all questions
        student_metadata: Dict with problem_set_id, gender, age, standard
        model: Which model's results to aggregate ("gemini" or "openai")
        
    Returns:
        StudentSummary with aggregated counts and per-question summaries
    """
    # Group transitions by question
    by_question: Dict[int, List[TransitionResult]] = defaultdict(list)
    for trans in all_transitions:
        qid = trans.question_id if trans.question_id is not None else 0
        by_question[qid].append(trans)
    
    # Aggregate each question
    question_summaries = []
    total_a = 0
    total_b = 0
    total_c = 0
    
    for qid in sorted(by_question.keys()):
        q_summary = aggregate_to_question(by_question[qid], model)
        question_summaries.append(q_summary)
        
        total_a += q_summary.misconception_a_count
        total_b += q_summary.misconception_b_count
        total_c += q_summary.misconception_c_count
    
    return StudentSummary(
        problem_set_id=student_metadata.get("problem_set_id", ""),
        gender=student_metadata.get("gender"),
        age=student_metadata.get("age"),
        standard=student_metadata.get("standard"),
        total_questions=len(question_summaries),
        total_transitions=len(all_transitions),
        misconception_a_total=total_a,
        misconception_b_total=total_b,
        misconception_c_total=total_c,
        questions=question_summaries
    )


def compare_models(transitions: List[TransitionResult]) -> Dict[str, Any]:
    """Compare Gemini and OpenAI results across transitions.
    
    Args:
        transitions: List of TransitionResult with both model results
        
    Returns:
        Dictionary with comparison statistics
    """
    total_with_both = 0
    full_agreement = 0
    a_agreement = 0
    b_agreement = 0
    c_agreement = 0
    
    gemini_a_total = 0
    gemini_b_total = 0
    gemini_c_total = 0
    
    openai_a_total = 0
    openai_b_total = 0
    openai_c_total = 0
    
    for trans in transitions:
        if trans.gemini is None or trans.openai is None:
            continue
        
        total_with_both += 1
        
        # Count detections
        if trans.gemini.misconception_a:
            gemini_a_total += 1
        if trans.gemini.misconception_b:
            gemini_b_total += 1
        if trans.gemini.misconception_c:
            gemini_c_total += 1
            
        if trans.openai.misconception_a:
            openai_a_total += 1
        if trans.openai.misconception_b:
            openai_b_total += 1
        if trans.openai.misconception_c:
            openai_c_total += 1
        
        # Check agreements
        if trans.gemini.misconception_a == trans.openai.misconception_a:
            a_agreement += 1
        if trans.gemini.misconception_b == trans.openai.misconception_b:
            b_agreement += 1
        if trans.gemini.misconception_c == trans.openai.misconception_c:
            c_agreement += 1
        
        if (trans.gemini.misconception_a == trans.openai.misconception_a and
            trans.gemini.misconception_b == trans.openai.misconception_b and
            trans.gemini.misconception_c == trans.openai.misconception_c):
            full_agreement += 1
    
    if total_with_both == 0:
        return {"error": "No transitions with both model results"}
    
    return {
        "total_transitions_compared": total_with_both,
        "full_agreement_rate": full_agreement / total_with_both,
        "misconception_a_agreement_rate": a_agreement / total_with_both,
        "misconception_b_agreement_rate": b_agreement / total_with_both,
        "misconception_c_agreement_rate": c_agreement / total_with_both,
        "gemini_detections": {
            "misconception_a": gemini_a_total,
            "misconception_b": gemini_b_total,
            "misconception_c": gemini_c_total
        },
        "openai_detections": {
            "misconception_a": openai_a_total,
            "misconception_b": openai_b_total,
            "misconception_c": openai_c_total
        }
    }

