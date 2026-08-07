# Changelog

## 2024-11-18 - Major Refactor

### ✅ Completed
- **Removed human label filtering** - Simplified workflow to just detection + analysis
- **Fixed schema issues** - Changed `question_id` from `int` to `str` to match data
- **Cleaned up repo** - Removed duplicate scripts, old configs, and unused files
- **Organized code** - All scripts now in `code/` with clear numbering:
  - `00_test_detection.py` - Quick test
  - `01_detect_transitions.py` - Main detection (API calls)
  - `02_analyze_results.py` - Analysis + visualization
- **Enhanced README** - Added emojis, better structure, clearer examples
- **Added USAGE.md** - Detailed guide with examples, tips, and troubleshooting

### 🔧 Technical Changes
- Updated `TransitionResult.question_id` from `Optional[int]` to `Optional[str]`
- Removed `temperature` parameter from OpenAI API calls (not supported by GPT-5)
- Fixed import issues (removed package-style imports within `code/`)
- Updated all prompts to use new 6-field schema (separate explanations per misconception)
- Removed empty `configs/` directory
- Cleaned up Python cache files

### 📊 Workflow
```
Input (transitions.feather)
    ↓
01_detect_transitions.py (LLM API calls)
    ↓
Output (results.feather)
    ↓
02_analyze_results.py
    ↓
Aggregations (question/student level) + Visualizations (3 PNGs)
```

### 🎯 Current Features
- Transition-level detection (both GPT-5 and Gemini)
- Question-level aggregation
- Student-level aggregation
- Model comparison & agreement analysis
- Rich visualizations:
  - Detection rates by model (bar chart)
  - Model agreement heatmap
  - Student-level distributions & correlations

### 📁 Final Structure
```
efm-misconceptions-pilot/
├── code/
│   ├── 00_test_detection.py
│   ├── 01_detect_transitions.py
│   ├── 02_analyze_results.py
│   ├── detector.py
│   ├── llm_clients.py
│   ├── schemas.py
│   ├── aggregation.py
│   ├── tabular.py
│   └── prompts/
│       ├── base.py
│       ├── misconception_a.j2
│       ├── misconception_b.j2
│       └── misconception_c.j2
├── data/
│   ├── linear_equations/
│   │   ├── transitions.feather
│   │   └── students.feather
│   └── detections/
├── README.md
├── USAGE.md
├── CHANGELOG.md
└── requirements.txt
```

### 🚀 Next Steps (if needed)
- [ ] Add batch processing for very large datasets
- [ ] Add cost tracking/estimation
- [ ] Add domain expansion beyond linear equations
- [ ] Add interactive dashboard (Streamlit/Plotly Dash)
- [ ] Add human label comparison (if labels available)

