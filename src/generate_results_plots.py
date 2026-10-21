import matplotlib.pyplot as plt

models = ["VQC", "QKSVM"]
accuracy = [0.9386, 0.5702]
f1 = [0.9536, 0.7066]
roc = [0.9861, 0.4778]

plt.figure(figsize=(8,5))
plt.bar(models, accuracy)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.savefig("accuracy_comparison.png", dpi=300)
plt.close()

plt.figure(figsize=(8,5))
plt.bar(models, f1)
plt.title("Model F1 Score Comparison")
plt.ylabel("F1 Score")
plt.savefig("f1_comparison.png", dpi=300)
plt.close()

plt.figure(figsize=(8,5))
plt.bar(models, roc)
plt.title("Model ROC-AUC Comparison")
plt.ylabel("ROC-AUC")
plt.savefig("roc_auc_comparison.png", dpi=300)
plt.close()

print("Plots generated successfully.")
