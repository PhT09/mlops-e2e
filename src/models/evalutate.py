import time
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and returns predictions and a dictionary of metrics.
    """
    t1 = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - t1
    
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    y_prob = model.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, y_prob, average="weighted", multi_class="ovr")
    
    metrics = {
        "pred_time": pred_time,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }
    
    return y_pred, metrics