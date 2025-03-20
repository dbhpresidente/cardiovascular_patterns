import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import itertools
import torch
from torchviz import make_dot
from torchview import draw_graph
import os
import pandas as pd

def plot_roc_curve(y_true_np, y_probs_np, output_dir, phase):
    roc_auc = roc_auc_score(y_true_np, y_probs_np)
    print(f"ROC AUC ({phase}): {roc_auc:.4f}")

    fpr, tpr, thresholds = roc_curve(y_true_np, y_probs_np)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel("Tasa de Falsos Positivos (FPR)")
    plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
    plt.title(f"Curva ROC ({phase})")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, f"roc_curve_{phase}.png"))
    plt.close()

def plot_confusion_matrix(y_true_np, y_pred_np, output_dir, phase):
    conf_matrix = confusion_matrix(y_true_np, y_pred_np)

    plt.figure(figsize=(6, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Matriz de Confusión ({phase})")
    plt.colorbar()

    classes = ["Clase 0", "Clase 1"]
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = conf_matrix.max() / 2
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, f"{conf_matrix[i, j]}", 
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

    plt.ylabel("Etiqueta Verdadera")
    plt.xlabel("Etiqueta Predicha")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{phase}.png"))
    plt.close()

def save_classification_report(y_true_np, y_pred_np, output_dir, phase):
    report = classification_report(y_true_np, y_pred_np, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_excel(os.path.join(output_dir, f"classification_report_{phase}.xlsx"))

def visualize_model(model, input_tensor, device, output_dir):
    input_tensor = input_tensor.to(device)
    make_dot(model(input_tensor), params=dict(model.named_parameters())).render(os.path.join(output_dir, "ecgcnn_architecture"), format="png")

    graph = draw_graph(model, input_tensor, expand_nested=True)
    graph.visual_graph.render(os.path.join(output_dir, "ecg_model_architecture"), format="png", cleanup=True)