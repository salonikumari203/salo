import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
N = 5000
age = np.random.randint(18, 65, size=N)
income = np.random.normal(50000, 15000, size=N).clip(20000, 100000)
visits = np.random.poisson(10, size=N)
time_on_site = np.random.exponential(scale=5, size=N)
clicked_ad = np.random.binomial(1, 0.3, size=N)

prob_purchase = (
    0.2 + 0.01*(age-30) + 0.00002*income + 0.03*clicked_ad + 0.01*visits
)
purchase = np.random.binomial(1, p=np.clip(prob_purchase, 0, 1))

df = pd.DataFrame({
    "age": age,
    "income": income,
    "visits": visits,
    "time_on_site": time_on_site,
    "clicked_ad": clicked_ad,
    "purchase": purchase
})

print("Dataset head:\n", df.head())
X = df.drop("purchase", axis=1)
y = df["purchase"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

for name, model, Xt in [
    ("Logistic Regression", log_reg, X_test_scaled),
    ("Random Forest", rf, X_test)
]:
    preds = model.predict(Xt)
    proba = model.predict_proba(Xt)[:,1]
    print(f"\nModel: {name}")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, proba))
    print(classification_report(y_test, preds))

feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
sns.barplot(x=feat_importances, y=feat_importances.index)
plt.title("Feature Importance (Random Forest)")
plt.show()
