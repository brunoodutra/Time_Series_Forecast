import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve

def plot_confusion_matrix(gesture_list = None, labels_val = None, labels_val_predicted= None,cf_matrix=None, Get_matrix=False) -> None:
    """
    Plots a confusion matrix using seaborn.

    Parameters:
    - gesture_list: List of gesture names
    - labels_val: True labels
    - labels_val_predicted: Predicted labels
    - cf_matrix: the computed confusion matrix
    - Get_matrix: If True, returns the confusion matrix as a NumPy array

    Returns:
    - If Get_matrix is True, returns the confusion matrix
    """
    if cf_matrix is None:
        cf_matrix = confusion_matrix(labels_val, labels_val_predicted)

    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]

    annotations = np.zeros(cf_matrix.shape)
    for i in range(len(annotations)):
        annotations[i, :] = np.round(cf_matrix[i, :] / np.sum(cf_matrix, axis=1)[i], 5)

    group_percentages = ["{0:.2%}".format(value) for value in annotations.flatten()]

    labels = [f"{v2}\n{v3}" for v2, v3 in zip(group_counts, group_percentages)]

    labels = np.asarray(labels).reshape(len(gesture_list), len(gesture_list))

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.set(font_scale=1.2)
    sns.heatmap(annotations, annot=labels, fmt='', cmap='Blues', cbar=True, xticklabels=gesture_list, yticklabels=gesture_list)

    plt.title('Confusion Matrix')
    
    if Get_matrix:
        return cf_matrix


def model_metrics(gesture_list, labels_val, labels_val_predicted, Get_metrics=False): 
    """
    Prints and optionally returns classification metrics.

    Parameters:
    - gesture_list: List of gesture names
    - labels_val: True labels
    - labels_val_predicted: Predicted labels
    - Get_metrics: If True, returns precision, recall, F1-Score, and accuracy

    Returns:
    - If Get_metrics is True, returns precision, recall, F1-Score, and accuracy
    """
    print(classification_report(labels_val, labels_val_predicted, target_names=[str(x) for x in gesture_list]))

    test_precision = precision_score(labels_val, labels_val_predicted, pos_label='positive', average='macro')
    print("Precision Score: ", test_precision)

    test_recall = recall_score(labels_val, labels_val_predicted, pos_label='positive', average='macro')
    print("Recall Score: ", test_recall)

    test_f1_score = f1_score(labels_val, labels_val_predicted, average='macro')
    print("F1-Score: ", test_f1_score)

    test_accuracy = accuracy_score(labels_val, labels_val_predicted)
    print("Accuracy: ", test_accuracy)
    
    if Get_metrics:
        return test_precision, test_recall, test_f1_score, test_accuracy
       

def model_average_std_metrics(labels=None, labels_predicted=None, cf_matrix=None, Get_metrics=True, Verbose=True): 
    """
    Calculates precision, recall, F1-Score, and accuracy for each class and provides mean and standard deviation.

    Parameters:
    - labels: True labels
    - labels_predicted: Predicted labels
    - cf_matrix: Confusion matrix
    - Get_metrics: If True, returns precision, recall, F1-Score, and accuracy along with mean and standard deviation
    - Verbose: If True, prints the mean and standard deviation

    Returns:
    - If Get_metrics is True, returns metrics for each class, mean and std for precision, recall, F1-Score, and accuracy
    """
    if cf_matrix is None:
        cf_matrix = confusion_matrix(labels, labels_predicted)

    precision_per_class = np.diag(cf_matrix) / np.sum(cf_matrix, axis=0)
    recall_per_class = np.diag(cf_matrix) / np.sum(cf_matrix, axis=1)
    f1_score_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class)
    accuracy_per_class = np.diag(cf_matrix) / np.sum(cf_matrix, axis=1)

    mean_precision = np.mean(precision_per_class)
    std_precision = np.std(precision_per_class)

    mean_recall = np.mean(recall_per_class)
    std_recall = np.std(recall_per_class)

    mean_f1_score = np.mean(f1_score_per_class)
    std_f1_score = np.std(f1_score_per_class)

    mean_accuracy = np.mean(accuracy_per_class)
    std_accuracy = np.std(accuracy_per_class)
    
    if Verbose:
        print(f'Precision Score: {np.round(mean_precision, 4)}±{np.round(std_precision, 4)}')
        print(f'Recall Score: {np.round(mean_recall, 4)}±{np.round(std_recall, 4)}')
        print(f'F1-Score: {np.round(mean_f1_score, 4)}±{np.round(std_f1_score, 4)}')
        print(f'Accuracy: {np.round(mean_accuracy, 4)}±{np.round(std_accuracy, 4)}')
    
    if Get_metrics:
        return [precision_per_class, recall_per_class, f1_score_per_class, accuracy_per_class], \
                [mean_precision, std_precision], \
                [mean_recall, std_recall], \
                [mean_f1_score, std_f1_score], \
                [mean_accuracy, std_accuracy]


def plot_metric_per_class(cf_matrix, class_labels, metric_name='Accuracy', method='per_class', figsize=(15, 6)):
    """
    Plots metrics (Accuracy, Precision, Recall, F1-Score) per class with standard deviation.

    Parameters:
    - cf_matrix: Confusion matrix
    - class_labels: List of class labels
    - metric_name: Name of the metric ('Accuracy', 'F1-Score', 'Recall', or 'Precision')
    - method: Method to calculate the metrics ('per_class' or 'All')
    - figsize: Figure size (optional)
    """
    metrics_per_class, pr, rec, f1, acc = model_average_std_metrics(cf_matrix=cf_matrix, Verbose=False) 

    if method == 'All':
        class_values = [pr[0], rec[0], f1[0], acc[0]]
        class_labels = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
        class_std = [pr[1], rec[1], f1[1], acc[1]]
        classes = class_labels
        group_label = 'Metrics'
        metric_name = 'Score'
    elif method == 'per_class':
        group_label = 'Classes'
        if metric_name not in ['Accuracy', 'F1-Score', 'Recall', 'Precision', 'All']:
            raise ValueError("Invalid metric_name. Choose from 'Accuracy', 'F1-Score', 'Recall', or 'Precision'.")
        # Calculate the metric for each class
        if metric_name == 'Accuracy':
            class_values = metrics_per_class[3]
        elif metric_name == 'F1-Score':
            class_values = metrics_per_class[2]
        elif metric_name == 'Recall':
            class_values = metrics_per_class[1]
        elif metric_name == 'Precision':
            class_values = metrics_per_class[0]  
        # Calculate the standard deviation or error for each class
        class_std = np.sqrt((class_values * (1 - class_values)) / np.sum(cf_matrix, axis=1))
        # Create a bar plot with error bars
        classes = [f'Class {i+1}' for i in range(len(class_values))]

    # Use Seaborn for a more stylish and colorful style
    sns.set(style="whitegrid", palette="muted")

    # Create a bar plot with error bars
    plt.figure(figsize=figsize)
    bars = plt.bar(classes, class_values, color=sns.color_palette("muted"), yerr=class_std, capsize=5, edgecolor="black")

    # Add labels to the bars
    for bar, value, std in zip(bars, class_values, class_std):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.30, f"{value:.3f} ± {std:.2f}", ha="center",
                 fontsize=9, color='black')

    # Add legend
    plt.legend(bars, class_labels, loc='upper left', bbox_to_anchor=(1, 1), title=group_label)

    plt.ylim(0, 1.01)  # Set the y-axis range from 0 to 1
    plt.xlabel(group_label)
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} per Class with Standard Deviation' if method == 'per_class' else f'Average {metric_name} with Standard Deviation')
    plt.show()

def get_roc_curve(labels, vals, predicted_vals, verbose=False, returnFig=False, figsize=(12, 10)):
    fig_roc_curve = plt.figure(1,figsize=figsize)
    auc_roc_vals = []
    for i in range(len(labels)):
        try:
                # Extracting only samples corresponding to class 0
                gt = (vals == i).astype(int)
                pred = (predicted_vals == i).astype(int) # Assuming your predictions are in a 2D array

                auc_roc = roc_auc_score(gt, pred)
                auc_roc_vals.append(auc_roc)
                
                if verbose:
                    print(f'ROC AUC for Class {i}: {auc_roc:.5f}')

                fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
                
                plt.plot([0, 1], [0, 1], 'k--')
                plt.plot(fpr_rf, tpr_rf,
                         label=str(labels[i]) + " (" + str(round(auc_roc, 3)) + ")")
                plt.xlabel('False positive rate')
                plt.ylabel('True positive rate')
                plt.title('ROC curve')
                plt.legend(loc='best')
        except:
                print(
                    f"Error in generating ROC curve for {labels[i]}. "
                    f"Dataset lacks enough examples."
                )

    plt.show()
    if returnFig:
        return auc_roc_vals, fig_roc_curve
    else:
        return auc_roc_vals

def calculate_classification_metrics(conf_mat,include_std=False):
    # Calculate precision, recall, F1-score, and accuracy for each class
    precision = np.diag(conf_mat) / np.sum(conf_mat, axis=0)
    recall = np.diag(conf_mat) / np.sum(conf_mat, axis=1)
    f1_score = 2 * (precision * recall) / (precision + recall)
    accuracy_per_class = np.diag(conf_mat) / np.sum(conf_mat, axis=1)

    # Calculate macro and weighted averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1_score = np.mean(f1_score)
    macro_accuracy = np.mean(accuracy_per_class)

    weighted_precision = np.sum(precision * np.sum(conf_mat, axis=1)) / np.sum(conf_mat)
    weighted_recall = np.sum(recall * np.sum(conf_mat, axis=1)) / np.sum(conf_mat)
    weighted_f1_score = np.sum(f1_score * np.sum(conf_mat, axis=1)) / np.sum(conf_mat)
    weighted_accuracy = np.sum(np.diag(conf_mat)) / np.sum(conf_mat)

    # Create a DataFrame with the results
    metrics_df = pd.DataFrame({
        'Class': range(len(precision)),
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score,
        'Accuracy': accuracy_per_class
    })

    # Add macro and weighted averages to the DataFrame
    metrics_df = metrics_df.append({
        'Class': 'Macro Avg',
        'Precision': macro_precision,
        'Recall': macro_recall,
        'F1-Score': macro_f1_score,
        'Accuracy': macro_accuracy
    }, ignore_index=True)

    if include_std:
        # Add standard deviation information to the DataFrame
        precision_std = np.std(precision)
        recall_std = np.std(recall)
        f1_score_std = np.std(f1_score)
        accuracy_std = np.std(accuracy_per_class)

        metrics_df = metrics_df.append({
            'Class': 'Std Macro Dev',
            'Precision': precision_std,
            'Recall': recall_std,
            'F1-Score': f1_score_std,
            'Accuracy': accuracy_std,
        }, ignore_index=True)
        
    metrics_df = metrics_df.append({
        'Class': 'Weighted Avg',
        'Precision': weighted_precision,
        'Recall': weighted_recall,
        'F1-Score': weighted_f1_score,
        'Accuracy': weighted_accuracy
    }, ignore_index=True)

    return metrics_df