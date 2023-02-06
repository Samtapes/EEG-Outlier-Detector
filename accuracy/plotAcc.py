import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('./accuracy/scores.csv')
milliseconds = df['Time_ms']
f1_scores = df['F1_Score']
cohen_scores = df['Cohen_Kappa']
roc_scores = df['AUC_Score']
traditional_scores = df['Accuracy']



plt.plot(milliseconds, f1_scores)
plt.xlabel("Milliseconds")
plt.ylabel("Score")
plt.title("F1 Score")
plt.grid(True)
plt.show()
plt.savefig("./accuracy/f1_scores.png")


plt.plot(milliseconds, cohen_scores, color='orange')
plt.xlabel("Milliseconds")
plt.ylabel("Score")
plt.title("Cohen Kappa")
plt.grid(True)
plt.show()
plt.savefig("./accuracy/cohen_kappas.png")

plt.plot(milliseconds, roc_scores, color='red')
plt.xlabel("Milliseconds")
plt.ylabel("Score")
plt.title("AUC")
plt.grid(True)
plt.show()
plt.savefig("./accuracy/auc_scores.png")

plt.plot(milliseconds, traditional_scores, color='green')
plt.xlabel("Milliseconds")
plt.ylabel("Score")
plt.title("Traditional Scores")
plt.grid(True)
plt.show()
plt.savefig("./accuracy/trad_scores.png")