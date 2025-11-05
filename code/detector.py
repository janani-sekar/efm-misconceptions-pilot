"""Main detector logic for analyzing student step transitions."""

import json
import warnings
from typing import Dict, List, Any, Optional
from llm_clients import GeminiClient, OpenAIClient
from prompts.base import create_detection_prompt, SYSTEM_PROMPT
from schemas import TransitionResult, TransitionDetection

# These misconceptions are specifically designed for linear equations
VALID_DOMAINS = ["linear_equations"]


class MisconceptionDetector:
    """Detect misconceptions in student work using LLMs with structured outputs."""
    
    def __init__(self, use_gemini: bool = True, use_openai: bool = True):
        """Initialize detector with LLM clients.
        
        Args:
            use_gemini: Whether to use Gemini API
            use_openai: Whether to use OpenAI API
        """
        self.gemini_client = GeminiClient() if use_gemini else None
        self.openai_client = OpenAIClient() if use_openai else None
        
        if not use_gemini and not use_openai:
            raise ValueError("At least one LLM client must be enabled")
    
    def detect_transition(
        self,
        step_from: str,
        step_to: str,
        question: Optional[str] = None,
        question_id: Optional[int] = None,
        transition_index: int = 0,
        model: str = "both",
        domain: Optional[str] = None,
        warn_invalid_domain: bool = True
    ) -> TransitionResult:
        """Detect misconceptions in a single step transition.
        
        Args:
            step_from: The starting step
            step_to: The next step
            question: Optional question context
            question_id: Optional question ID
            transition_index: Index of this transition within the question
            model: Which model to use ("gemini", "openai", or "both")
            domain: Optional domain (e.g., "linear_equations", "algebraic_expressions", "exponents")
            warn_invalid_domain: Whether to warn if domain is not valid for these misconceptions
            
        Returns:
            TransitionResult with structured detection results
        """
        # Check domain validity
        if domain and domain not in VALID_DOMAINS and warn_invalid_domain:
            warnings.warn(
                f"Domain '{domain}' is not in valid domains {VALID_DOMAINS}. "
                f"These misconceptions are designed for linear equations only. "
                f"Results may not be meaningful.",
                UserWarning
            )
        prompt = create_detection_prompt(step_from, step_to, question)
        
        result = TransitionResult(
            step_from=step_from,
            step_to=step_to,
            question=question,
            question_id=question_id,
            transition_index=transition_index
        )
        
        # Get Gemini results
        if model in ["gemini", "both"] and self.gemini_client:
            try:
                gemini_detection = self.gemini_client.generate_structured(
                    prompt,
                    SYSTEM_PROMPT,
                    TransitionDetection
                )
                result.gemini = gemini_detection
            except Exception as e:
                result.gemini_error = str(e)
        
        # Get OpenAI results
        if model in ["openai", "both"] and self.openai_client:
            try:
                openai_detection = self.openai_client.generate_structured(
                    prompt,
                    SYSTEM_PROMPT,
                    TransitionDetection
                )
                result.openai = openai_detection
            except Exception as e:
                result.openai_error = str(e)
        
        return result
    
    def detect_question(
        self,
        question_data: Dict[str, Any],
        model: str = "both"
    ) -> List[TransitionResult]:
        """Detect misconceptions across all transitions in a question.
        
        Args:
            question_data: Question dictionary with 'question', 'steps', 'answer'
            model: Which model to use ("gemini", "openai", or "both")
            
        Returns:
            List of TransitionResult for each transition
        """
        steps = question_data.get("steps", [])
        question_text = question_data.get("question", "")
        question_id = question_data.get("question_id")
        
        # Need at least 2 steps to have a transition
        if len(steps) < 2:
            return []
        
        results = []
        for i in range(len(steps) - 1):
            step_from = steps[i].get("step", "")
            step_to = steps[i + 1].get("step", "")
            
            # Skip if either step is empty or crossed out
            if not step_from or not step_to:
                continue
            if steps[i].get("crossed_out", False) or steps[i + 1].get("crossed_out", False):
                continue
            
            transition_result = self.detect_transition(
                step_from=step_from,
                step_to=step_to,
                question=question_text,
                question_id=question_id,
                transition_index=i,
                model=model
            )
            results.append(transition_result)
        
        return results
    
    def detect_student(
        self,
        student_data: Dict[str, Any],
        model: str = "both"
    ) -> Dict[str, Any]:
        """Detect misconceptions across all questions for a student.
        
        Args:
            student_data: Full student JSON data
            model: Which model to use ("gemini", "openai", or "both")
            
        Returns:
            Dictionary with student metadata and all detection results
        """
        questions = student_data.get("questions", [])
        
        # Collect all transitions
        all_transitions = []
        for question in questions:
            question_detections = self.detect_question(question, model)
            all_transitions.extend(question_detections)
        
        # Build result dictionary
        result = {
            "problem_set_id": student_data.get("problem_set_id"),
            "gender": student_data.get("gender"),
            "age": student_data.get("age"),
            "standard": student_data.get("standard"),
            "total_transitions": len(all_transitions),
            "transitions": [t.model_dump() for t in all_transitions]
        }
        
        return result
