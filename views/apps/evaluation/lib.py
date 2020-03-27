""" Evaluation library. """
import pandas as pd  # type: ignore

from sklearn import metrics  # type: ignore


def mean_squared_error(actuals: pd.Series, preds: pd.Series) -> float:
    """Computes MSE given array of actuals and probs."""
    return metrics.mean_squared_error(y_true=actuals, y_pred=preds)
