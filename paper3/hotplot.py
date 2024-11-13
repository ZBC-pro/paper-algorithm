import matplotlib.pyplot as plt
import pandas as pd

# 定义模型名称列表和数据框
models = ['Model1', 'Model2', 'Model3', 'Model4', 'Model5', 'Model6', 'Model7']
data = {
    'Model': models,
    'Accuracy': [0.671, 0.622, 0.681, 0.694, 0.717, 0.833, 0.875],
    'Precision': [0.665, 0.624, 0.674, 0.688, 0.709, 0.843, 0.865],
    'Recall': [0.671, 0.622, 0.681, 0.693, 0.715, 0.847, 0.867],
    'F1-Score': [0.643, 0.623, 0.675, 0.689, 0.713, 0.843, 0.866]
}
df = pd.DataFrame(data)
df.set_index('Model', inplace=True)

# 绘制河流图（堆叠面积图）
plt.figure(figsize=(10, 6))
plt.stackplot(models, df.T, labels=df.columns, alpha=0.8)
plt.legend(loc='upper left')
plt.title("Performance Comparison of Different Models (Streamgraph)")
plt.xlabel("Models")
plt.ylabel("Score")
plt.show()