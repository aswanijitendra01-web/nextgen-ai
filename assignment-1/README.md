# Diabetes Prediction using Machine Learning

End-to-end binary classification pipeline that predicts whether a patient is likely to have diabetes from medical attributes, using **Logistic Regression** in Python and Scikit-learn.

- **Dataset:** `diabetes.csv` (Pima Indians Diabetes Dataset, 768 patients, 8 features)
- **Target:** `Outcome` (0 = No Diabetes, 1 = Diabetes)
- **Notebook:** [`Diabetes_Prediction_Assignment1.ipynb`](Diabetes_Prediction_Assignment1.ipynb) (all code, outputs, plots and explanations)
- **Assignment brief:** [`Assignment-1.pdf`](Assignment-1.pdf)

## Workflow

| Part | What was done |
|---|---|
| 1. Data preparation | Loaded data, EDA (shape, dtypes, missing values, statistics, correlations), feature/target separation, stratified 80/20 train-test split, preprocessing |
| 2. Model building | Logistic Regression, with justification for choosing it |
| 3. Model training | Training accuracy; test accuracy, precision, recall, F1, confusion matrix; train vs test comparison |
| 4. Performance analysis | Baseline comparison, ROC-AUC, cross-validation, coefficient/odds-ratio analysis, threshold tuning, conclusions |

## Key preprocessing decisions

- **Hidden missing values:** `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` and `BMI` contain `0` values that are physiologically impossible (e.g. 374 zeros in `Insulin`). They were converted to `NaN` and imputed with the **median**.
- **Feature engineering:** BMI category, age group (one-hot encoded) and a Glucose x BMI interaction.
- **Scaling:** `StandardScaler`.
- **No data leakage:** imputer and scaler are fitted on the training set only.
- **Split:** 614 training / 154 testing samples, stratified (class ratio preserved).

## Results

| Metric | Training | Testing |
|---|---|---|
| Accuracy | 78.8% | 73.4% |
| Precision | 74.7% | 64.4% |
| Recall | 59.4% | 53.7% |
| F1-score | 66.2% | 58.6% |
| ROC-AUC | n/a | 0.841 |

- 5-fold cross-validated accuracy: **77.0% +/- 2.7%**
- Majority-class baseline accuracy: 64.9%
- Test confusion matrix: TN = 84, FP = 16, FN = 25, TP = 29

### Main findings

1. **Glucose** is by far the strongest predictor, followed by BMI, age, pedigree function and pregnancies.
2. The model generalises reasonably well (mild ~5 point train/test accuracy gap) and clearly beats the baseline.
3. At the default 0.5 threshold, recall for diabetic patients is low (about 54%). Because ROC-AUC is high, lowering the threshold to 0.3 raises recall to about 83% (false negatives drop from 25 to 9), which is more suitable for a screening use case. (Threshold was explored on the test set for illustration only.)

<p align="center">
  <img src="images/confusion_matrix.png" width="340"> <img src="images/roc_curve.png" width="380">
</p>

## Run it yourself

```bash
git clone <your-repo-url>
cd diabetes-prediction-ml
pip install -r requirements.txt
jupyter notebook Diabetes_Prediction_Assignment1.ipynb
```

`diabetes.csv` must be in the same folder as the notebook. The trained model and preprocessor are saved in `diabetes_logreg_model.joblib` (load with `joblib.load`).

## Limitations

- Small dataset (768 rows; only 54 diabetic cases in the test set), so metrics are noisy.
- Dataset covers only female patients of Pima Indian heritage aged 21+, so results may not generalise to other populations.
- This is an educational model and not a diagnostic tool.

## Possible improvements

Hyperparameter tuning (`C`, `class_weight`), non-linear models (Random Forest, Gradient Boosting, SVM), KNN/iterative imputation, and threshold selection on a validation set.
