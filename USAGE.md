# Usage Guide

## Quick Example

```bash
# Activate your venv
source .venv/bin/activate

# Test detection works
python code/00_test_detection.py

# Run detection on 50 transitions
python code/01_detect_transitions.py \
  --transitions-file data/tabular/combined/transitions.feather \
  --output-file data/detections/my_results.feather \
  --limit 50 \
  --model both

# Analyze and visualize
python code/02_analyze_results.py data/detections/my_results.feather \
  --output-dir results/ \
  --save-aggregations
```

## Input Data Format

Your transitions file should be a `.feather` or `.csv` with these columns:

| Column | Type | Description | Required |
|--------|------|-------------|----------|
| `transition_id` | int/str | Unique identifier | ✅ |
| `student_id` | str | Student identifier | ✅ |
| `question_id` | str | Question identifier | ✅ |
| `step_text` | str | Current step | ✅ |
| `step_text_next` | str | Next step | ✅ |
| `question_text` | str | Question prompt | Optional |
| `implied` | int | Mathematical correctness (0/1) | Optional |

## Output Data Format

Results saved as `.feather` with:

### Per Model (gemini_* and openai_*)
- `{model}_misconception_a` (bool)
- `{model}_explanation_a` (str)
- `{model}_misconception_b` (bool)
- `{model}_explanation_b` (str)
- `{model}_misconception_c` (bool)
- `{model}_explanation_c` (str)

### Metadata (preserved from input)
- `transition_id`, `student_id`, `question_id`
- `step_from`, `step_to`, `question`
- `implied` (mathematical correctness if provided)

### Error Tracking
- `gemini_error` (str, null if successful)
- `openai_error` (str, null if successful)

## Aggregation Levels

### Transition Level
Raw LLM outputs with one row per step transition.

### Question Level
Aggregates all transitions within a question:
- Count/rate of each misconception per question
- Number of transitions per question
- Correctness stats per question

### Student Level
Aggregates all transitions across all questions for each student:
- Total/rate of each misconception per student
- Number of questions and transitions per student
- Overall correctness rate per student

## Visualizations

### 1. Detection Rates (`detection_rates.png`)
Bar chart comparing Gemini vs OpenAI detection rates for each misconception type.

### 2. Model Agreement (`model_agreement.png`)
Heatmap showing:
- Individual model detection rates
- Agreement percentage between models
- Per misconception type

### 3. Student Analysis (`student_analysis.png`)
4-panel figure:
- **Top Left:** Distribution of transitions per student
- **Top Right:** Distribution of questions per student
- **Bottom Left:** Distribution of misconception rates across students
- **Bottom Right:** Scatter plot of correctness vs misconception rate (with trend line)

## Tips & Tricks

### Rate Limiting
If you hit rate limits, reduce `--max-workers`:
```bash
python code/01_detect_transitions.py ... --max-workers 2
```

### Testing
Always test with `--limit` first:
```bash
# Test with just 10 transitions
python code/01_detect_transitions.py ... --limit 10
```

### Single Model
To save costs, test with just one model:
```bash
# Just OpenAI
python code/01_detect_transitions.py ... --model openai

# Just Gemini
python code/01_detect_transitions.py ... --model gemini
```

### Domain Check
The detector automatically filters to `linear_equations` domain. To disable the warning:
```python
# In detector.py, set warn_invalid_domain=False in detect_transition()
result = detector.detect_transition(..., warn_invalid_domain=False)
```

## Cost Estimates

Rough estimates (as of 2024):

- **GPT-5:** ~$0.001-0.003 per transition
- **Gemini 2.5 Flash:** ~$0.0001-0.0003 per transition

For 1000 transitions with `--model both`:
- Total: ~$1.10-3.30
- Recommend testing with `--limit 100` first

## Troubleshooting

### API Key Issues
```bash
# Check if .env is loaded
cat .env

# Test keys work
python code/00_test_detection.py
```

### Import Errors
```bash
# Make sure you're in venv
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Empty Results
Check that your transitions file has the required columns:
```python
import pandas as pd
df = pd.read_feather("your_file.feather")
print(df.columns.tolist())
```

### Model Timeouts
Increase timeout or reduce workers:
```bash
python code/01_detect_transitions.py ... --max-workers 1
```

