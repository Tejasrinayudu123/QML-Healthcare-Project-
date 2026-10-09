import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_pca(X_pca, y):
    plt.figure(figsize=(7,5))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='coolwarm', alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Projection of WDBC Dataset")
    plt.savefig("pca_plot.png", dpi=300)
    plt.close()

def plot_roc(y_test, y_proba, name):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend()
    plt.savefig(f"roc_{name}.png", dpi=300)
    plt.close()
