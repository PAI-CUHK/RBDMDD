import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc)


def plot_roc_curves(results_df, y_true):
    plt.figure(figsize=(10, 8))
    for _, row in results_df.iterrows():
        fpr, tpr, _ = roc_curve(y_true, row['y_probas'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{row['Model']} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - MDDonly vs MDDRBD')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(OUTPUT_DIR / "plots" / "mdd_rbd_ROC_Curves.png", dpi=300)
    plt.close()