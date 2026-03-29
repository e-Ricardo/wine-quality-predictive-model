# 🍷 Wine Quality Prediction — Deep Learning Project

**Team:** Lei Zhang · Xiangdong Luo · Preethi Nanjunda Swamy  
**Course:** Machine Learning Final Project  
**Dataset:** [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)

---

## 📁 Project Structure

```
wine_quality_project/
├── data/                   # Raw & processed data (auto-downloaded)
├── figures/                # All saved plots & charts
├── models/                 # Saved trained model files (.keras)
├── src/
│   ├── 01_eda.py           # Step 1: Exploratory Data Analysis
│   ├── 02_preprocessing.py # Step 2: Data cleaning, SMOTE, splitting
│   ├── 03_models.py        # Step 3: MLP, 1D-CNN, LSTM definitions
│   ├── 04_train.py         # Step 4: Train all models
│   ├── 05_evaluate.py      # Step 5: Metrics, confusion matrices, curves
│   └── 06_feature_analysis.py  # Step 6: PCA, SHAP, feature importance
├── utils/
│   └── helpers.py          # Shared utility functions
├── main.py                 # ▶️ Run entire pipeline at once
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone / open in VS Code
```bash
code wine_quality_project
```

### 2. Create virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run full pipeline
```bash
python main.py
```

### 5. OR run step by step
```bash
python src/01_eda.py
python src/02_preprocessing.py
python src/03_models.py        # just prints model summaries
python src/04_train.py
python src/05_evaluate.py
python src/06_feature_analysis.py
```

---

## 📊 Models Implemented

| Model    | Architecture                          | Purpose                        |
|----------|---------------------------------------|-------------------------------|
| MLP      | Dense(256→128→64→3)                   | Baseline feedforward network  |
| 1D-CNN   | Conv1D(64→128→64) + Dense(128→3)      | Local feature correlations    |
| LSTM     | LSTM(128→64) + Dense(64→3)            | Sequential feature dependency |
| XGBoost  | Gradient boosted trees (hybrid)       | Classical ML comparison       |

---

## 📈 Output Files

After running, check:
- `figures/` — EDA plots, training curves, confusion matrices, SHAP plots
- `models/` — Saved `.keras` model files
- `data/` — Processed numpy arrays

---

## 📋 Task Allocation

| Member | Responsibility |
|---|---|
| Lei Zhang | Model design, hyperparameter tuning, final evaluation |
| Xiangdong Luo | EDA, preprocessing, SMOTE, feature selection |
| Preethi Nanjunda Swamy | Model implementation (MLP/CNN/LSTM), training, result visualization |
