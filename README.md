# Customer Churn Prediction (Telco)

Small end-to-end project where I try to predict customer churn for a telecom company.

I’m an 18-year-old Data Science student and this project helped me practice a full
supervised learning workflow: from cleaning the data to training and comparing models.

---

## 📂 Dataset

- **Source:** Telco Customer Churn dataset (customers from a telecom company).
- **Rows:** Each row is one customer.
- **Target variable:** `Churn` (0 = stays, 1 = leaves).
- **Features:** Contract type, monthly charges, tenure (months as a customer),
  services like internet, phone, streaming, security, etc.

---

## 🎯 Project goals

- Understand which type of customers are more likely to leave.
- Build a simple baseline model using **Logistic Regression**.
- Train a more powerful model using **Random Forest**.
- Compare both models and see which one works better.
- Visualize results in a clear way (heatmaps, ROC curve, confusion matrix, feature importance).

---

## 🛠 Tech stack

- **Python**
- **Pandas, NumPy**
- **Matplotlib, Seaborn**
- **Scikit-learn** (Logistic Regression, Random Forest, metrics)

---

## 🧹 Data preparation

Main steps:

1. **Load data** from `telco_churn.csv`.
2. Convert `TotalCharges` to numeric and fill missing values with the median.
3. Drop `customerID` because it is only an identifier.
4. Encode categorical features using `LabelEncoder`.
5. Split the data into **train** and **test** sets (80% / 20%).
6. Scale the features (only for Logistic Regression) using `StandardScaler`.

---

## 📊 Exploratory Data Analysis (EDA)

Some of the things I checked:

- Distribution of **Churn** (the classes are imbalanced: more customers stay than leave).
- Churn by **gender** and **contract type**.
- Relationship between **tenure** and churn: new customers tend to leave more.
- Correlation matrix between numerical features.

All the plots are saved in the `plots/` folder.

---

## 🤖 Models

### 1. Logistic Regression

- Trained on **scaled features**.
- Good baseline model.
- Accuracy on test set: ~**0.80**.
- Works especially well for predicting customers who **stay** (class 0).

### 2. Random Forest

- Trained on the original (non-scaled) features.
- Accuracy on test set: ~**0.79**.
- Similar performance to Logistic Regression, but more complex.

### Feature Importance (Random Forest)

The most important features for predicting churn are:

- **Contract type**
- **Tenure** (months as a customer)
- **MonthlyCharges**
- Services like **OnlineSecurity**, **TechSupport**, etc.

This matches common sense: customers with short tenure, month-to-month contracts and
higher charges tend to churn more.

---

## 📈 Evaluation visuals

I created a few plots to understand the models better:

- **ROC Curve** for Logistic Regression (`plots/roc_curve_logistic.png`)
- **Confusion Matrix** for Logistic Regression (`plots/confusion_matrix_logistic.png`)
- **Feature importance** for Random Forest (`plots/feature_importance_rf.png`)
- **Correlation heatmap** (`plots/heatmap_correlation.png`)

---

## ✅ Main takeaways (as a junior data scientist)

- Logistic Regression already gives a strong baseline (~80% accuracy).
- Random Forest performs very similar on this dataset.
- Tenure, contract type and monthly charges are key drivers of churn.
- Class imbalance makes it harder to detect customers who will leave.
- This project helped me practice a full ML workflow: EDA → preprocessing → modeling → evaluation.

---

## 🚀 Next steps

Things I would like to try in the future:

- Handle class imbalance (SMOTE, class weights).
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV).
- Try other models like XGBoost or Gradient Boosting.
- Deploy a simple API or dashboard to score new customers.
