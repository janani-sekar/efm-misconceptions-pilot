# EFM Misconceptions Pilot

Automated detection of common algebraic misconceptions in student work using LLMs.

## Setup

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up API keys
echo "OPENAI_API_KEY=your_key" > .env
echo "GEMINI_API_KEY=your_key" >> .env
```

## Workflow

### Step 0: Filter to Human-Labeled Data

```bash
python code/00_filter_to_human_labels.py
```

**Output:** 27 students, 1,505 transitions → `data/linear_equations/`

### Step 1: Test Detection (Optional)

```bash
python code/01_test_detection.py
```

### Step 2: Run LLM Detection (MAKES API CALLS - COSTS MONEY)

```bash
# Test with 20 transitions first
python code/02_detect_all_transitions.py \
  --transitions-file data/linear_equations/transitions.feather \
  --output-file data/detections/human_labeled.feather \
  --limit 20 \
  --model openai

# Full run on all 1,505 transitions
python code/02_detect_all_transitions.py \
  --transitions-file data/linear_equations/transitions.feather \
  --output-file data/detections/human_labeled.feather \
  --model both \
  --max-workers 8
```

### Step 3: Analyze Results

```bash
python code/03_analyze_results.py data/detections/human_labeled.jsonl
```

## Misconceptions

**A) Moving Terms Without Changing Signs**
- From: `3x + 5 = 2x + 10`
- To: `3x + 2x = 5 + 10` ❌

**B) Distributive Property Error**
- From: `2(3x + 4) = 5x - 1`
- To: `6x + 4 = 5x - 1` ❌

**C) Combining Unlike Terms**
- From: `3x + 5 = 2x + 10`
- To: `8x = 12x` ❌

## Project Structure

```
efm-misconceptions-pilot/
├── code/
│   ├── 00_filter_to_human_labels.py  # Filter to labeled data
│   ├── 01_test_detection.py          # Test with examples
│   ├── 02_detect_all_transitions.py  # Run LLM detection
│   ├── 03_analyze_results.py         # Analyze results
│   ├── detector.py                   # Helper
│   ├── llm_clients.py                # Helper
│   ├── schemas.py                    # Helper
│   ├── aggregation.py                # Helper
│   ├── tabular.py                    # Helper
│   └── prompts/                      # Prompt templates
├── data/
│   ├── misconceptions_apo_v1.csv    # Human labels (input)
│   ├── tabular/convergent/linear_equations/ # Source
│   ├── linear_equations/            # Filtered (output of step 0)
│   └── detections/                  # LLM results (output of step 2)
├── .env
├── requirements.txt
└── README.md
```

## Output Format

**Structured output with 6 fields per transition:**
- `misconception_a` + `explanation_a`
- `misconception_b` + `explanation_b`  
- `misconception_c` + `explanation_c`
- Plus: `implied` (mathematical correctness from source data)

**Models:** GPT-5 and Gemini 2.5 Flash
