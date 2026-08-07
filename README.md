# EFM Misconceptions Pilot

Automated detection of common algebraic misconceptions in student step transitions using LLMs (GPT-5 & Gemini).

## 🚀 Quick Start

```bash
# 1. Set up environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Add API keys
echo "OPENAI_API_KEY=your_key" > .env
echo "GEMINI_API_KEY=your_key" >> .env

# 3. Test it works
python code/00_test_detection.py

# 4. Run detection on your transitions (MAKES API CALLS)
python code/01_detect_transitions.py \
  --transitions-file data/tabular/combined/transitions.feather \
  --output-file data/detections/results.feather \
  --limit 20 \
  --model both

# 5. Analyze and visualize
python code/02_analyze_results.py data/detections/results.feather \
  --output-dir results/ \
  --save-aggregations
```

## 📊 Workflow

### 0️⃣ Test Detection (Optional)

```bash
python code/00_test_detection.py
```

Runs quick test on example transitions to verify API keys and LLM clients work.

### 1️⃣ Run LLM Detection ⚠️ **MAKES API CALLS**

```bash
# Start with a small test
python code/01_detect_transitions.py \
  --transitions-file data/tabular/combined/transitions.feather \
  --output-file data/detections/results.feather \
  --limit 20 \
  --model both

# Full run with parallelization
python code/01_detect_transitions.py \
  --transitions-file data/tabular/combined/transitions.feather \
  --output-file data/detections/results.feather \
  --model both \
  --max-workers 8
```

**Required input columns in transitions file:**
- `step_text` - Current step
- `step_text_next` - Next step  
- `question_text` - Question prompt (optional)
- `question_id` - Question identifier
- `student_id` - Student identifier
- `transition_id` - Unique transition ID
- `implied` - Mathematical correctness (optional, 0/1)

**CLI Options:**
- `--model`: `openai`, `gemini`, or `both` (default: `both`)
- `--limit`: Process only first N transitions (for testing)
- `--max-workers`: Number of parallel API calls (default: 4)

### 2️⃣ Analyze & Visualize

```bash
python code/02_analyze_results.py data/detections/results.feather \
  --output-dir results/ \
  --save-aggregations
```

**Console Output:**
- Transition-level summary (detection rates, model agreement)
- Question-level aggregation (avg transitions per question)
- Student-level aggregation (avg questions/transitions per student)
- Mathematical correctness stats

**Visual Outputs:**
- `detection_rates.png` - Detection rates by model (bar chart)
- `model_agreement.png` - Model comparison heatmap
- `student_analysis.png` - 4-panel student distributions & correlations

**Data Outputs** (with `--save-aggregations`):
- `question_aggregates.csv` - Question-level aggregated results
- `student_aggregates.csv` - Student-level aggregated results

## 🎯 Misconceptions Detected

### A) Moving Terms Without Changing Signs
**Description:** Student moves a term across the equals sign but forgets to change its sign.

**Example:**
```
From: 3x + 5 = 2x + 10
To:   3x + 2x = 5 + 10  ❌ Should be: 3x - 2x = 10 - 5
```

### B) Distributive Property Error
**Description:** Student incorrectly applies the distributive property, often forgetting to multiply all terms inside parentheses.

**Example:**
```
From: 2(3x + 4) = 5x - 1
To:   6x + 4 = 5x - 1   ❌ Should be: 6x + 8 = 5x - 1
```

### C) Combining Unlike Terms
**Description:** Student combines terms that cannot be combined (different variables or mixing variables with constants).

**Example:**
```
From: 3x + 5 = 2x + 10
To:   8x = 12x          ❌ Invalid combination
```

## Project Structure

```
efm-misconceptions-pilot/
├── code/
│   ├── 00_test_detection.py         # Test with examples
│   ├── 01_detect_transitions.py     # Run LLM detection (API CALLS)
│   ├── 02_analyze_results.py        # Analyze + visualize
│   ├── detector.py                  # Detection logic
│   ├── llm_clients.py               # OpenAI & Gemini clients
│   ├── schemas.py                   # Pydantic output schemas
│   ├── aggregation.py               # Question/student aggregation
│   ├── tabular.py                   # Data prep helpers
│   └── prompts/                     # Prompt templates (.j2 files)
├── data/
│   ├── tabular/combined/
│   │   └── transitions.feather      # Input: transition data
│   └── detections/                  # Output: LLM results
├── results/                         # Output: plots & aggregates
├── .env                             # API keys
├── requirements.txt
└── README.md
```

## 📤 Output Schema

Each transition gets analyzed by both models with **6 structured fields**:

```json
{
  "misconception_a": true/false,
  "explanation_a": "Why misconception A is/isn't present",
  "misconception_b": true/false,
  "explanation_b": "Why misconception B is/isn't present",
  "misconception_c": true/false,
  "explanation_c": "Why misconception C is/isn't present"
}
```

Results saved with columns:
- `openai_misconception_a`, `openai_explanation_a`, ...
- `gemini_misconception_a`, `gemini_explanation_a`, ...
- `implied` - Mathematical correctness from source data (0/1)

**Models:** GPT-5 (OpenAI) and Gemini 2.5 Flash (Google)

## 💡 Key Features

- **Parallel Processing:** Concurrent API calls for faster analysis
- **Structured Outputs:** Type-safe Pydantic schemas enforced via API
- **Multi-Model:** Compare GPT-5 vs Gemini detection patterns
- **Multi-Level:** Aggregation from transition → question → student
- **Rich Visualizations:** Distribution plots, heatmaps, correlation analysis
- **Modular Prompts:** Each misconception has its own Jinja2 template

## 🔧 Notes

- Only works on **linear equations** domain (enforced by detector)
- Requires `implied` column for correctness correlation plots
- Detections run in parallel (be mindful of API rate limits)
- Results flattened automatically for easy pandas analysis

---

📖 **For detailed usage, see [USAGE.md](USAGE.md)**
