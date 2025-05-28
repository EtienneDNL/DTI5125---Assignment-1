# 📊 Text Classification of Scientific Abstracts Using Sentence Embeddings and Logistic Regression

## 🔍 Project Overview

This project explores **text classification** of scientific abstracts using a variety of **pretrained sentence embedding models** and **classification algorithms**.

The task focuses on categorizing abstracts into **five AI-related topics** based on their semantic content.  
It is a **group assignment** for the course **DTI/GNG 5125 at the University of Ottawa**.

---

## 📁 Repository Structure

| Folder/File                         | Description                                          |
|-------------------------------------|------------------------------------------------------|
| `Data/`                             | Contains raw and processed abstract data             |
| `Embedding Models/`                 | Bar charts comparing model accuracies and F1 scores  |
| `hyperParameter_tuning/`            | Hyperparameter tuning outputs                        |
| `roc_curves/`                       | Saved ROC curve images for each model                |
| `scipy-env/`                        | Virtual environment setup for dependencies           |
| `shap_outputs/`                     | SHAP explainability outputs (optional)               |
| `data_preparation.ipynb`            | Data loading, cleaning, and partitioning             |
| `data_transformation.ipynb`         | Embedding generation, model training, evaluation     |
| `results_visualization.ipynb`       | Bar chart generation and result visualization        |
| `main.py`                           | Optional Python script for automation                |
| `README.md`                         | This project overview file                           |
| `pyproject.toml`                    | Project metadata for `uv` package manager            |
| `uv.lock`                           | Dependency lock file                                 |

---

## 📚 Data Description

- **Source**: Abstracts from 5 AI-related research domains:
  - AI and Supply Chain Management
  - AI and Finance
  - AI and Marketing
  - AI and Economics
  - AI and Accounting
- **Preprocessing**:
  - Cleaning text (stopwords removal, tokenization)
  - Splitting abstracts into ~100-word partitions
  - Labeling per topic

---

## 🧠 Classifier Models Used

We tested the following **classifier models**:
- **Logistic Regression** (main focus of analysis)
- **Naive Bayes (Multinomial and Gaussian)** (for baseline comparisons)
- **Support Vector Machine (SVM)** (with various kernels)
- **Cross-validation** (10-fold) used for model evaluation

---

## 🧠 Embedding Models Tested

- `jinaai/jina-embeddings-v3`
- `all-MiniLM-L6-v2`
- `all-mpnet-base`
- `paraphrase-MiniLM-L3-v2`
- `allenai/specter2`

---

## 🔧 Hyperparameter Tuning

We explored the impact of key Logistic Regression hyperparameters:
- `C` (regularization strength)
- `solver` (optimization algorithm)


Bar charts were generated to visualize the effect of these parameters on model accuracy.

---

## 📊 Results Summary

| Embedding Model                  | Accuracy | Precision | Recall  | F1 Score |
|---------------------------------|----------|-----------|---------|----------|
| **JinaAI v3**                   | 0.822    | 0.826     | 0.822   | 0.822    |
| **all-miniLm-L6-v2**            | 0.803    | 0.804     | 0.803   | 0.803    |
| **all-mpnet-base**              | 0.809    | 0.810     | 0.809   | 0.808    |
| **paraphrase-MiniLM-L3-v2**     | 0.746    | 0.742     | 0.746   | 0.744    |
| **allenai/specter2**            | 0.802    | 0.802     | 0.802   | 0.802    |

✅ **Best overall performance**: `JinaAI v3`.  
✅ **ROC curves and bar charts** provide clear visual comparisons.

---

## 📈 Visualizations

- **ROC curves** for each embedding model (saved in `roc_curves/`).
- **Bar charts** for accuracy and F1 scores (saved in `Embedding Models/`).

---

## 💡 Insights

- **Embedding choice significantly impacts classification performance**.
- Larger, domain-specific models (e.g., `mpnet`, `jina-v3`) outperform lighter models.
- Logistic Regression, while simple, performs well when paired with strong embeddings.
- **Class 3** consistently showed lower AUC scores—highlighting a potential area for error analysis.

---




