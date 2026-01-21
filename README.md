# Boosting for Imbalanced Datasets with XGBoost

## 📌 Project Overview
Handling imbalanced datasets is a critical challenge in machine learning. This project demonstrates how to use **XGBoost** combined with **SMOTE** (Synthetic Minority Over-sampling Technique) to improve classification performance on datasets where the minority class is significantly underrepresented.

## 🎯 Objectives
* **Implement XGBoost:** Leverage gradient boosting for high-performance classification.
* **Handle Imbalance:** Use SMOTE and `scale_pos_weight` to prevent model bias toward the majority class.
* **Hyperparameter Tuning:** Optimize the model using `RandomizedSearchCV`.
* **Robust Evaluation:** Shift focus from Accuracy to Precision-Recall curves and F1-Scores.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** `XGBoost`, `Scikit-learn`, `Imbalanced-learn` (SMOTE), `Pandas`, `Matplotlib`, `Seaborn`
* **Environment:** Jupyter Notebook

## 📊 Methodology
1. **Data Preprocessing:** Splitting data into stratified training and testing sets.
2. **Resampling:** Applying SMOTE within a cross-validation pipeline to avoid data leakage.
3. **Training:** Configuring `XGBClassifier` with class weight adjustments.
4. **Optimization:** Tuning `max_depth`, `learning_rate`, and `n_estimators`.
5. **Evaluation:** Generating Confusion Matrices and Precision-Recall Curves.

## 📈 Key Results
* **Imbalance Mitigation:** Successfully increased the Recall of the minority class without devastating Precision.
* **ROC-AUC vs PR-AUC:** Demonstrated why Precision-Recall AUC is more informative for imbalanced tasks than standard ROC curves.

## 🚀 How to Use
1. **Clone the repo:**
   ```bash
   git clone [https://github.com/VikashSenthilKumar/xgboost-imbalance-project.git](https://github.com/VikashSenthilKumar/xgboost-imbalance-project.git)
