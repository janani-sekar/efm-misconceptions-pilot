"""Base prompt components and builders."""

from pathlib import Path

# Load prompt templates from .j2 files
_PROMPTS_DIR = Path(__file__).parent

with open(_PROMPTS_DIR / "misconception_a.j2") as f:
    MISCONCEPTION_A_DESCRIPTION = f.read()

with open(_PROMPTS_DIR / "misconception_b.j2") as f:
    MISCONCEPTION_B_DESCRIPTION = f.read()

with open(_PROMPTS_DIR / "misconception_c.j2") as f:
    MISCONCEPTION_C_DESCRIPTION = f.read()


SYSTEM_PROMPT = """You are an expert mathematics educator specializing in identifying common algebraic misconceptions in student work.

Analyze step transitions in student work carefully and accurately. You will determine if specific misconceptions are present when a student moves from one step to the next.

Always respond with valid JSON matching the required schema."""


def create_detection_prompt(step_from: str, step_to: str, question: str = None) -> str:
    """Create a comprehensive prompt for detecting all three misconceptions.
    
    Args:
        step_from: The starting step
        step_to: The next step
        question: Optional question context
        
    Returns:
        Formatted prompt string
    """
    context = ""
    if question:
        context = f"**Original Problem:** {question}\n\n"
    
    prompt = f"""You are analyzing a single transition in student work on linear equations.

{context}**Step 1:** {step_from}
**Step 2:** {step_to}

---

# Misconceptions to Detect

{MISCONCEPTION_A_DESCRIPTION}

{MISCONCEPTION_B_DESCRIPTION}

{MISCONCEPTION_C_DESCRIPTION}

---

# Your Task

Analyze the transition from Step 1 to Step 2 and determine:

1. Is **Misconception A** (moving terms without changing signs) present?
2. Is **Misconception B** (distributive property error) present?
3. Is **Misconception C** (combining unlike terms) present?

For each misconception, respond with `true` if present, `false` if not present.

Provide a separate explanation for EACH misconception.

Respond in JSON format with the following structure:
{{
  "misconception_a": true/false,
  "explanation_a": "Explanation for why misconception A is or is not present",
  "misconception_b": true/false,
  "explanation_b": "Explanation for why misconception B is or is not present",
  "misconception_c": true/false,
  "explanation_c": "Explanation for why misconception C is or is not present"
}}"""
    
    return prompt


def create_single_misconception_prompt(
    misconception_type: str,
    step_from: str,
    step_to: str,
    question: str = None
) -> str:
    """Create a prompt for detecting a single specific misconception.
    
    Args:
        misconception_type: "a", "b", or "c"
        step_from: The starting step
        step_to: The next step
        question: Optional question context
        
    Returns:
        Formatted prompt string
    """
    descriptions = {
        "a": MISCONCEPTION_A_DESCRIPTION,
        "b": MISCONCEPTION_B_DESCRIPTION,
        "c": MISCONCEPTION_C_DESCRIPTION
    }
    
    names = {
        "a": "Moving Terms Without Changing Signs",
        "b": "Distributive Property Error",
        "c": "Combining Unlike Terms"
    }
    
    if misconception_type.lower() not in descriptions:
        raise ValueError(f"Unknown misconception type: {misconception_type}")
    
    context = ""
    if question:
        context = f"**Original Problem:** {question}\n\n"
    
    prompt = f"""You are analyzing a single transition in student work.

{context}**Step 1:** {step_from}
**Step 2:** {step_to}

---

# Misconception: {names[misconception_type.lower()]}

{descriptions[misconception_type.lower()]}

---

# Your Task

Analyze the transition from Step 1 to Step 2 and determine:

**Is this misconception present?**

Respond with:
- `true` if the misconception is present
- `false` if the misconception is NOT present

Provide a brief explanation of your reasoning.

Respond in JSON format:
{{
  "present": true/false,
  "explanation": "Your reasoning here"
}}"""
    
    return prompt

