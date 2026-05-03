import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sentence = ["I", "love", "deep", "learning"]

attention_weights = np.array([0.1, 0.3, 0.4, 0.2])

# Softmax
attention_weights = np.exp(attention_weights) / np.sum(np.exp(attention_weights))

sns.heatmap([attention_weights], annot=True, xticklabels=sentence, cmap="Blues")

plt.title("Attention Heatmap")
plt.show()