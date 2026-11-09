from sklearn.metrics import classification_report

def print_final_results(y_true, y_pred):
    print(classification_report(y_true, y_pred))