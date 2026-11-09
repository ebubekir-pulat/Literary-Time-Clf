from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

def print_final_results(y_true, y_pred):
    print(classification_report(y_true, y_pred))
    
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.show()