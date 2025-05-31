import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def eval(model,X_valid,y_val_encoded):
    model_pred = model.predict(X_valid)
    model_pred_class = np.argmax(model_pred, axis= 1)
    model_results = evaluate(y_true= y_val_encoded, y_pred= model_pred_class)

    return model_results

def evaluate(y_true,y_pred):
    """ Calculate models accuracy, precesion, recall and f1score """
    # Basic Metrics
    print("Accuracy: in %", (accuracy_score(y_true, y_pred)*100))
    print("Precision:", precision_score(y_true, y_pred, average='weighted'))
    print("Recall:", recall_score(y_true, y_pred, average='weighted'))
    print("F1 Score:", f1_score(y_true, y_pred, average='weighted'))

    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
