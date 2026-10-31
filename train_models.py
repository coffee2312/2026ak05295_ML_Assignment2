"""Train 5 classifiers, save models, test_data.csv, metrics + robustness tables."""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from common import TARGET, add_signal_features, add_noise, evaluate

SEED = 7
df = pd.read_csv("prepared_data.csv")
X, y = df.drop(columns=[TARGET]), df[TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

def make(clf):
    return Pipeline([("signal", FunctionTransformer(add_signal_features)),
                     ("scale", StandardScaler()), ("clf", clf)])

models = {
    "Logistic Regression": make(LogisticRegression(max_iter=2000)),
    "Decision Tree": make(DecisionTreeClassifier(max_depth=15, random_state=SEED)),
    "KNN": make(KNeighborsClassifier(n_neighbors=5)),
    "Naive Bayes": make(GaussianNB()),
    "Support Vector Machine": make(SVC(kernel="rbf", probability=True, random_state=SEED)),
}

os.makedirs("model", exist_ok=True)
rows, robust = [], []
for name, m in models.items():
    m.fit(X_tr, y_tr)
    joblib.dump(m, f"model/{name.lower().replace(' ', '_')}.joblib", compress=3)   # compress: KNN stores its whole training set
    rows.append({"ML Model Name": name, **evaluate(m, X_te, y_te)})
    for sigma in (0, 3, 6, 9):                       # robustness to RSSI noise
        Xn = pd.DataFrame(add_noise(X_te, sigma, drop_frac=0.1 if sigma else 0, seed=SEED),
                          columns=X_te.columns)
        robust.append({"ML Model Name": name, "noise_sigma_dB": sigma,
                       "Accuracy": evaluate(m, Xn, y_te)["Accuracy"]})
    print("done", name)

pd.concat([X_te, y_te], axis=1).to_csv("test_data.csv", index=False)
pd.DataFrame(rows).round(4).to_csv("metrics.csv", index=False)
pd.DataFrame(robust).round(4).to_csv("robustness.csv", index=False)
print("Models saved in model/. Scores written to metrics.csv and robustness.csv (not printed).\nOpen the Streamlit app and upload test_data.csv to see them.")
