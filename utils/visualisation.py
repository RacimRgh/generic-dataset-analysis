import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class RegressionPlot:
    def predict_regression_plot(y_true, y_pred):
        fig, ax = plt.subplots()
        ax.scatter(y_true, y_pred)
        ax.plot([y_true.min(), y_true.max()], [
                y_true.min(), y_true.max()], 'k--', lw=4)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        ax.set_title('Mean Squared Error')
        ax.set_xlim(y_true.min(), y_true.max())
        ax.set_ylim(y_true.min(), y_true.max())
        mse = mean_squared_error(y_true, y_pred)
        ax.text(0.05, 0.95, 'MSE = {:.2f}'.format(mse),
                transform=ax.transAxes, va='top')
        return fig


class ClassificationPlot:
    def plot_confusion_matrix(y_true, y_pred, classes):
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        cm_display = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=classes)
        fig = cm_display.plot(include_values=True, xticks_rotation='vertical')
        return fig, ax


def plot_metrics(x_test, y_test):

    st.subheader("Confusion Matrix")
    plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
    st.pyplot()

    st.subheader("ROC Curve")
    plot_roc_curve(model, x_test, y_test)
    st.pyplot()

    st.subheader("Precision-Recall Curve")
    plot_precision_recall_curve(model, x_test, y_test)
    st.pyplot()
