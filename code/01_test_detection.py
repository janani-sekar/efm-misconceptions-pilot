#!/usr/bin/env python3
"""Test misconception detection with example transitions.

MAKES API CALLS to OpenAI/Gemini.
"""

from detector import MisconceptionDetector

# Example transitions
test_cases = [
    {
        "name": "Misconception A: Sign error when moving terms",
        "step_from": "3x + 5 = 2x + 10",
        "step_to": "3x + 2x = 5 + 10",
        "question": "Solve: 3x + 5 = 2x + 10",
        "expected": {"a": True, "b": False, "c": False}
    },
    {
        "name": "Misconception B: Distributive property error",
        "step_from": "2(3x + 4) = 5x - 1",
        "step_to": "6x + 4 = 5x - 1",
        "question": "Solve: 2(3x + 4) = 5x - 1",
        "expected": {"a": False, "b": True, "c": False}
    },
    {
        "name": "Misconception C: Combining unlike terms",
        "step_from": "3x + 5 = 2x + 10",
        "step_to": "8x = 12x",
        "question": "Solve: 3x + 5 = 2x + 10",
        "expected": {"a": False, "b": False, "c": True}
    },
    {
        "name": "Correct transition (no misconceptions)",
        "step_from": "3x + 5 = 2x + 10",
        "step_to": "3x - 2x = 10 - 5",
        "question": "Solve: 3x + 5 = 2x + 10",
        "expected": {"a": False, "b": False, "c": False}
    }
]


def main():
    print("Testing Misconception Detection")
    print("=" * 70)
    
    # Initialize detector
    try:
        detector = MisconceptionDetector(use_gemini=True, use_openai=True)
        print("✓ Detector initialized\n")
    except ValueError as e:
        print(f"✗ Error: {e}")
        print("Make sure .env file has API keys!")
        return
    
    # Test each case
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 70)
        print(f"From: {test_case['step_from']}")
        print(f"To:   {test_case['step_to']}")
        
        try:
            result = detector.detect_transition(
                step_from=test_case["step_from"],
                step_to=test_case["step_to"],
                question=test_case["question"],
                transition_index=0,
                model="both"
            )
            
            # Display OpenAI results
            if result.openai and not result.openai_error:
                detection = result.openai
                print(f"\nOpenAI (GPT-5) Results:")
                print(f"  A: {detection.misconception_a}")
                print(f"     {detection.explanation_a}")
                print(f"  B: {detection.misconception_b}")
                print(f"     {detection.explanation_b}")
                print(f"  C: {detection.misconception_c}")
                print(f"     {detection.explanation_c}")
                
                expected = test_case["expected"]
                correct = (
                    detection.misconception_a == expected["a"] and
                    detection.misconception_b == expected["b"] and
                    detection.misconception_c == expected["c"]
                )
                print(f"\n  {'✓ PASS' if correct else '✗ FAIL'}")
            
            elif result.openai_error:
                print(f"\n  OpenAI Error: {result.openai_error}")
            
            # Display Gemini results
            if result.gemini and not result.gemini_error:
                detection = result.gemini
                print(f"\nGemini Results:")
                print(f"  A: {detection.misconception_a}")
                print(f"     {detection.explanation_a}")
                print(f"  B: {detection.misconception_b}")
                print(f"     {detection.explanation_b}")
                print(f"  C: {detection.misconception_c}")
                print(f"     {detection.explanation_c}")
            
            elif result.gemini_error:
                print(f"\n  Gemini Error: {result.gemini_error}")
                
        except Exception as e:
            print(f"\n  Exception: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 70)
    
    print("\n" + "=" * 70)
    print("Testing Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

