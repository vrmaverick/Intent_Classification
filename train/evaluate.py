import numpy as np

def eval(model,X_valid,y_val_encoded):
    model_pred = model.predict(X_valid)
    model_pred_class = np.argmax(model_4_pred, axis= 1)
    model_results = evaluate(y_true= y_val_encoded, y_pred= model_pred_class)

    return model_results