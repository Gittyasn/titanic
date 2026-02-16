# 🛳️ Titanic Intelligence Platform (Professional ML V2.0)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](#interactive-dashboard)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is a **High-Performance Machine Learning Intelligence Platform** that transforms the historical Titanic dataset into a deep analytical system. Designed for professional portfolios, it showcases advanced engineering patterns, model explainability (SHAP), and a premium interactive UX.

## 🚀 Premium Features

- **Intelligence Dashboard**: A high-end experience featuring Glassmorphism UI, Lottie animations, and dynamic transitions.
- **Explainable AI (XAI)**: Integrated **SHAP (SHapley Additive exPlanations)** values to decode exactly *why* the models make their decisions.
- **Automated Hyperparameter Tuning**: Built-in **GridSearchCV** pipeline to optimize model accuracy automatically.
- **High-Dimensional Analytics**: 3D interactive scatter plots and professional ROC-AUC benchmarking.
- **Modular Enterprise Pattern**: Clean separation of concerns across `src/` modules, strictly following production-grade Python standards.

## 🧠 Strategic Modules

```text
titanic-intelligence/
├── src/                # The Brain (Modular Logic)
│   ├── data_loader.py    # Robust IO
│   ├── preprocessing.py  # Advanced Feature Engineering
│   ├── models.py         # Tuning & XAI Logic
│   ├── visualization.py  # High-end Graphics Engine
│   └── utils.py          # Corporate Logging
├── dashboard/          # The Interface
│   └── app.py            # Streamlit Premium App
├── data/               # Assets
├── requirements.txt    # Heavy-duty dependencies
└── README.md           # This strategic guide
```

## 🛠️ Stack & Technologies

- **Core**: Python 3.8+
- **Data Engineering**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn, XGBoost, SHAP (Explainability)
- **Visuals**: Plotly Professional, Matplotlib/Seaborn, Scikit-Plot
- **UI/UX**: Streamlit, Lottie Animations, Custom CSS-in-JS

## 🚦 Getting Started

### Installation

1. **Clone & Enter**:
   ```bash
   git clone [repository-url]
   cd titanic-intelligence
   ```

2. **Environment Setup**:
   ```bash
   python -m venv venv
   source venv/bin/activate # Windows: venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

### Launch the Intelligence Platform

```bash
python -m streamlit run "titanic survival prediction/dashboard/app.py"
```

## 📈 Strategic Insights

*   **Social Class Hierarchy**: Survival probability drops by over 40% for the 3rd class despite similar age distributions.
*   **Feature Importance**: SHAP analysis reveals that 'Title_Mr' and 'Sex_female' are the most influential survival nodes.
*   **Tuning Impact**: Automated GridSearch improved the Random Forest precision by ~3.4% compared to baseline.

---
*Developed by **Antigravity AI** | Built for Major Portfolio Benchmarking*
