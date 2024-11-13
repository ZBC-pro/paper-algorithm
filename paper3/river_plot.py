import matplotlib.pyplot as plt
import pandas as pd

# 创建数据框，根据用户上传的表格内容
models = ['LR', 'DT', 'SVM', 'RNN', 'LSTM', 'CNN', 'Our HMM Model']
data = {
    'Accuracy': [0.671, 0.622, 0.681, 0.694, 0.717, 0.833, 0.875],
    'Precision': [0.665, 0.624, 0.674, 0.688, 0.709, 0.843, 0.865],
    'Recall': [0.671, 0.622, 0.681, 0.693, 0.715, 0.847, 0.867],
    'F1-Score': [0.643, 0.623, 0.675, 0.689, 0.713, 0.843, 0.866]
}
df = pd.DataFrame(data, index=models)

# 绘制河流图（堆叠面积图）
plt.figure(figsize=(10, 6))
plt.stackplot(df.index, df.T, labels=df.columns, alpha=0.8)
plt.legend(loc='upper left')
plt.title("Performance Comparison of Different Models (Streamgraph)")
plt.xlabel("Models")
plt.ylabel("Score")
plt.show()