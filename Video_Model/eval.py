import numpy as np
import matplotlib.pyplot as plt

"""
Ground truth labels are expected to be one-hot encoded.
For example, for binary classification:
- [0, 1] -> Fake (1)
- [1, 0] -> Real (0)

Predicted labels are expected to be class probabilities.
For example, for binary classification:
- [0.2, 0.8] -> 80% confidence of Fake (1)
- [0.7, 0.3] -> 70% confidence of Real (0)
"""

# Confusion Matrix

def confusion_matrix(y_true : np.ndarray, y_pred : np.ndarray, class_idx : int = None, threshold : float = 0.5) -> np.ndarray:
    """
    Calculate the confusion matrix.

    Parameters:
    y_true (np.ndarray): Array of true labels (one-hot) (B, C).
    y_pred (np.ndarray): Array of predicted labels (class probs) (B, C).
    class_idx (int, optional): Index of the class to filter by. If None, computes for all classes.
    threshold (float, optional): Threshold for binary classification. Default is 0.5.

    Returns:
    np.ndarray: Confusion matrix of shape (num_classes, num_classes) if class_idx is None.
    np.ndarray: Confusion matrix of shape (2, 2) if class_idx is specified.
        [[TN, FP],
         [FN, TP]]
    """
    num_classes = y_true.shape[-1]
    cm = np.zeros((num_classes, num_classes), dtype=int)
    if class_idx is None:
        for i in range(len(y_true)):
            true_label = np.argmax(y_true[i])
            pred_label = np.argmax(y_pred[i])
            cm[true_label, pred_label] += 1

            return cm

    assert class_idx < num_classes, "class_idx must be less than the number of classes"
    assert threshold >= 0 and threshold <= 1, "threshold must be between 0 and 1"

    y_pred_binary = (y_pred >= threshold).astype(int)
    for i in range(len(y_true)):
        true_label = y_true[i, class_idx].astype(int)
        pred_label = y_pred_binary[i, class_idx].astype(int)
        cm[true_label, pred_label] += 1

    cm_filtered = np.zeros((2, 2), dtype=int)
    cm_filtered[0, 0] = cm[0, 0]  # True Positives
    cm_filtered[0, 1] = np.sum(cm[0, :]) - cm_filtered[0, 0]  # False Negatives
    cm_filtered[1, 0] = np.sum(cm[:, 0]) - cm_filtered[0, 0]  # False Positives
    cm_filtered[1, 1] = np.sum(cm) - cm_filtered[0, 0] - cm_filtered[0, 1] - cm_filtered[1, 0]  # True Negatives
    return cm_filtered

def metric_history(metric_function, y_true : np.ndarray, y_pred : np.ndarray, class_idx : int = None, num_samples : int = 100, include_threshold=False) -> list:
    """
    Calculate the history of a metric over a range of thresholds between 0 (inclusive) and 1 (inclusive).

    Parameters:
    metric_function (callable): Function to calculate the metric.
    y_true (np.ndarray): Array of true labels (one-hot) (B, C).
    y_pred (np.ndarray): Array of predicted labels (class scores) (B, C).
    class_idx (int, optional): Index of the class to filter by. If None, computes for all classes (if possible).
    num_samples (int, optional): Number of samples to compute the metric for. Default is 100.

    Returns:
    list: List of metric values for each threshold.
        If include_threshold is True, returns a list of tuples (metric_value, threshold).
        If include_threshold is False, returns a list of metric values.
    """
    if include_threshold:
        return [(metric_function(y_true, y_pred, class_idx=class_idx, threshold=threshold), threshold) for threshold in np.linspace(0.0, 1.0, num_samples)]
    else:
        return [metric_function(y_true, y_pred, class_idx=class_idx, threshold=threshold) for threshold in np.linspace(0.0, 1.0, num_samples)]

# Accuracy (ACC)

def accuracy(y_true : np.ndarray, y_pred : np.ndarray, class_idx : int = None, threshold : float = 0.5) -> float:
    """
    Calculate the accuracy of predictions.
    If class_idx is given, calculates the accuracy for the class given by class_idx. (TP + TN) / Total.

    Parameters:
    y_true (np.ndarray): Array of true labels (one-hot) (B, C).
    y_pred (np.ndarray): Array of predicted labels (class probs) (B, C).
    class_idx (int, optional): Index of the class to compute accuracy for. If None, computes overall accuracy.
    threshold (float, optional): Threshold for binary classification. Default is 0.5.

    Returns:
    float: Accuracy as a percentage.
    """
    if class_idx is None:
        return np.mean(y_true == y_pred) * 100

    cm = confusion_matrix(y_true, y_pred, class_idx=class_idx, threshold=threshold)
    true_positives = cm[1, 1]  # True Positives
    true_negatives = cm[0, 0]  # True Negatives
    total = np.sum(cm)
    accuracy = (true_positives + true_negatives) / total
    return accuracy * 100

##########

# Precision and Recall

def precision(y_true : np.ndarray, y_pred : np.ndarray, class_idx : int, threshold : float = 0.5) -> float:
    """
    Calculate the precision of predictions for the class given by class_idx. TP / (TP + FP).

    Parameters:
    y_true (np.ndarray): Array of true labels (one-hot) (B, C).
    y_pred (np.ndarray): Array of predicted labels (class probs) (B, C).
    class_idx (int): Index of the class to compute precision for.
    threshold (float, optional): Threshold for binary classification. Default is 0.5.

    Returns:
    float: Precision as a percentage.
    """
    cm = confusion_matrix(y_true, y_pred, class_idx=class_idx, threshold=threshold)
    true_positives = cm[1, 1]  # True Positives
    false_positives = cm[0, 1]  # False Positives
    precision = true_positives / (true_positives + false_positives + 1e-10)  # Adding a small value to avoid division by zero
    return precision * 100

def recall(y_true : np.ndarray, y_pred : np.ndarray, class_idx : int, threshold : float = 0.5) -> float:
    """
    Calculate the recall of predictions for the class given by class_idx. TP / (TP + FN).

    Parameters:
    y_true (np.ndarray): Array of true labels (one-hot) (B, C).
    y_pred (np.ndarray): Array of predicted labels (class probs) (B, C).
    class_idx (int): Index of the class to compute recall for.
    threshold (float, optional): Threshold for binary classification. Default is 0.5.

    Returns:
    float: Recall as a percentage.
    """
    cm = confusion_matrix(y_true, y_pred, class_idx=class_idx, threshold=threshold)
    true_positives = cm[1, 1]  # True Positives
    false_negatives = cm[1, 0]  # False Negatives
    recall = true_positives / (true_positives + false_negatives + 1e-10)  # Adding a small value to avoid division by zero
    return recall * 100

# Average Precision (AP)

def average_precision(y_true : np.ndarray, y_pred : np.ndarray, class_idx : int, num_samples : int = 100) -> float:
    """
    Calculate the average precision for multi-class predictions.

    Parameters:
    y_true (np.ndarray): Array of true labels (one-hot) (B, C).
    y_pred (np.ndarray): Array of predicted labels (class scores) (B, C).
    class_idx (int): Index of the class to compute average precision for.

    Returns:
    float: Average precision as a percentage.
    """
    return sum(metric_history(precision, y_true, y_pred, class_idx=class_idx, num_samples=num_samples)) / num_samples

##########

# F1-Score

def f1_score(y_true : np.ndarray, y_pred : np.ndarray, class_idx : int, threshold : float = 0.5) -> float:
    """
    Calculate the F1 score of predictions for the class given by class_idx. 2 * (Precision * Recall) / (Precision + Recall).

    Parameters:
    y_true (np.ndarray): Array of true labels (one-hot) (B, C).
    y_pred (np.ndarray): Array of predicted labels (class probs) (B, C).
    class_idx (int): Index of the class to compute F1 score for.
    threshold (float, optional): Threshold for binary classification. Default is 0.5.

    Returns:
    float: F1 score as a percentage.
    """
    p = precision(y_true, y_pred, class_idx=class_idx, threshold=threshold)
    r = recall(y_true, y_pred, class_idx=class_idx, threshold=threshold)
    f1 = 2 * (p * r) / (p + r + 1e-10)  # Adding a small value to avoid division by zero
    return f1

##########

# Receiver Operating Characteristic (ROC) and Area Under the Curve (AUC) (ROC-AUC)

def roc_curve(y_true : np.ndarray, y_pred : np.ndarray, class_idx : int, num_samples : int = 100, sorted=True) -> tuple:
    """
    Calculate the ROC curve for multi-class predictions.

    Parameters:
    y_true (np.ndarray): Array of true labels (one-hot) (B, C).
    y_pred (np.ndarray): Array of predicted labels (class scores) (B, C).
    class_idx (int): Index of the class to compute ROC curve for.
    num_samples (int, optional): Number of samples to compute the ROC curve for. Default is 100.
    sorted (bool, optional): If True, sorts the points by false positive rate. Default is True.

    Returns:
    tuple: Tuple containing:
        - fpr (np.ndarray): False positive rates.
        - tpr (np.ndarray): True positive rates.
        - thresholds (np.ndarray): Thresholds used to compute the ROC curve.
    """
    thresholds = np.linspace(0.0, 1.0, num_samples)
    fpr = []
    tpr = []

    for threshold in thresholds:
        cm = confusion_matrix(y_true, y_pred, class_idx=class_idx, threshold=threshold)
        true_positives = cm[1, 1]
        false_positives = cm[0, 1]
        false_negatives = cm[1, 0]
        true_negatives = cm[0, 0]

        fpr.append(false_positives / (false_positives + true_negatives + 1e-10))  # Adding a small value to avoid division by zero
        tpr.append(true_positives / (true_positives + false_negatives + 1e-10))  # Adding a small value to avoid division by zero

    fpr = np.array(fpr)
    tpr = np.array(tpr)

    if sorted:
        sort = np.argsort(fpr)
        fpr_sort, tpr_sort, thresholds_sort = fpr[sort], tpr[sort], thresholds[sort]
        _, sort_unique = np.unique(fpr_sort, return_index=True)  # Remove duplicates
        return fpr_sort[sort_unique], tpr_sort[sort_unique], thresholds_sort[sort_unique]

    return fpr, tpr, thresholds

def roc_auc(y_true : np.ndarray, y_pred : np.ndarray, class_idx : int, num_samples : int = 100) -> float:
    """
    Calculate the ROC AUC for multi-class predictions.

    Parameters:
    y_true (np.ndarray): Array of true labels (one-hot) (B, C).
    y_pred (np.ndarray): Array of predicted labels (class scores) (B, C).
    class_idx (int): Index of the class to compute ROC AUC for.
    num_samples (int, optional): Number of samples to compute the ROC AUC for. Default is 100.

    Returns:
    float: ROC AUC as a percentage.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred, class_idx=class_idx, num_samples=num_samples)
    return np.trapezoid(tpr, fpr) * 100

##########

# Equal Error Rate (EER)

def equal_error_rate(y_true : np.ndarray, y_pred : np.ndarray, class_idx : int, num_samples : int = 100) -> float:
    """
    Calculate the Equal Error Rate (EER) for multi-class predictions.
    EER is the point where the false positive rate equals the false negative rate.

    Parameters:
    y_true (np.ndarray): Array of true labels (one-hot) (B, C).
    y_pred (np.ndarray): Array of predicted labels (class scores) (B, C).
    class_idx (int): Index of the class to compute EER for.
    num_samples (int, optional): Number of samples to compute the EER for. Default is 100.

    Returns:
    float: Equal Error Rate as a percentage.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred, class_idx=class_idx, num_samples=num_samples)
    fnr = 1 - tpr  # False Negative Rate
    eer_index = np.argmin(np.abs(fpr - fnr))
    return 100 * (fpr[eer_index] + fnr[eer_index]) / 2