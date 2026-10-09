from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from .metrics import evaluate_model

def run_classical_baselines(X_train, y_train, X_test, y_test):
    reports = []

    svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
    reports.append(
        evaluate_model(
            svm, X_train, y_train, X_test, y_test,
            name="RBF-SVM",
            notes="Standard RBF-SVM baseline."
        )
    )

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
    )
    reports.append(
        evaluate_model(
            rf, X_train, y_train, X_test, y_test,
            name="Random Forest",
            notes="Strong tabular baseline."
        )
    )

    return reports
