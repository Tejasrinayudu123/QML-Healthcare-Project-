import time
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

@dataclass
class ModelReport:
    name: str
    accuracy: float
    f1: float
    roc_auc: float
    train_time_sec: float
    infer_time_sec: float
    n_qubits: int
    circuit_depth: int
    n_shots: int
    notes: str = ""

def evaluate_model(model, X_train, y_train, X_test, y_test,
                   n_qubits=0, circuit_depth=0, n_shots=0, name="", notes=""):
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train

    start_infer = time.time()
    y_pred = model.predict(X_test)
    infer_time = time.time() - start_infer

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = getattr(model, "decision_function", lambda x: y_pred)(X_test)
        roc = roc_auc_score(y_test, y_proba)
    except Exception:
        roc = float("nan")

    return ModelReport(
        name=name,
        accuracy=acc,
        f1=f1,
        roc_auc=roc,
        train_time_sec=train_time,
        infer_time_sec=infer_time,
        n_qubits=n_qubits,
        circuit_depth=circuit_depth,
        n_shots=n_shots,
        notes=notes,
    )
