import itertools
import os
import numpy as np

from scipy.io import wavfile
from python_speech_features import mfcc, logfbank, delta
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from hmmlearn import hmm

import matplotlib
import matplotlib.pyplot as plt
import pickle
import json

from matplotlib import rcParams
from matplotlib import font_manager

import time

# 找到系统中支持中文的字体
font_path = "/System/Library/Fonts/Supplemental/Songti.ttc"  # 替换为系统中支持的字体路径
custom_font = font_manager.FontProperties(fname=font_path)

# 设置字体为全局默认字体
rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()

# 查看支持的字体列表
# print(matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf'))


# 数据预处理
# 拆分训练集和测试集
# 建立字典
def build_dataset(sound_path):
    files = sorted(os.listdir(sound_path))
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    data = {}
    i = 0

    for f in files:
        # 读取每一个音频文件 f 就是刚才在spoken_digit中的所有语音文件
        # /Users/dususu/Desktop/data/spoken_digit + 0_george_0.wav 拼接所有的音频文件
        feature = feature_extractor(sound_path=sound_path + '/' + f)
        # lab = int(f[0])
        if i % 5 == 0:
            x_test.append(feature.tolist())  # 转为 Python 列表
            y_test.append(int(f[0]))
        else:
            x_train.append(feature.tolist())  # 转为 Python 列表
            y_train.append(int(f[0]))  # 修正标签为整型
        i += 1

    # 将训练数据按类别分组，并存储到字典中，k 是 label，v 是 label 的特征数据。 划分为10组
    for i in range(0, len(x_train), len(x_train) // 10):
        data[y_train[i]] = x_train[i:i + len(x_train) // 10]

    return x_train, y_train, x_test, y_test, data


# 步骤二：提取特征
# 提取MFCC特征，sound_path 声音路径
def feature_extractor(sound_path):
    sampling_freq, audio = wavfile.read(sound_path)
    mfcc_features = mfcc(audio, sampling_freq)
    return mfcc_features


# 步骤三：模型训练
# 处理 Nan 值
imp = SimpleImputer(strategy="mean")

# 训练模型
def train_model(data):
    learned_hmm = {}    # 每个数字的模型都存为一个字典
    for label in data.keys():
        # 隐藏状态、混合高斯个数
        model = hmm.GMMHMM(n_components=4, n_mix=3, covariance_type="diag", n_iter=1000)
        feature = np.ndarray(shape=(1, 13))
        feature = imp.fit_transform(feature)
        for list_feature in data[label]:
            feature = np.vstack((feature, list_feature))
        obj = model.fit(feature)
        learned_hmm[label] = obj

    return learned_hmm


# GMM-HMM 模型的参数保存为 JSON 文件
def save_hmm_models_to_json(learned_hmm, file_name):

    serializable_hmm = {}

    for label, model in learned_hmm.items():
        serializable_hmm[label] = {
            "startprob": model.startprob_.tolist(), # 初始状态概率分布
            "transmat": model.transmat_.tolist(),   # 状态转移矩阵
            "means": model.means_.tolist(),     # 高斯混合分布的均值
            "covars": model.covars_.tolist(),   # 高斯混合分布的协方差
            "weights": model.weights_.tolist()      # 高斯混合分布的权重
        }

    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(serializable_hmm, f, indent=4)
    print(f"HMM 模型参数已保存到 {file_name}")


# 步骤四：测试
def prediction(test_data, trained):
    # 测试数据的预测结果
    predict_label = []

    if (type(test_data) == type([])):   # 如果测试数据是一组
        for test in test_data:
            scores = []
            for node in trained.keys():
                scores.append(trained[node].score(test))    # test在当前模型文件的预测得分
            predict_label.append(scores.index(max(scores)))     # 最大得分的成员
    else:  # 只有一个元素
        scores = []
        for node in trained.keys():
            scores.append(trained[node].score(node))    # test在当前模型文件的预测得分
        predict_label.append(scores.index(max(scores)))     # 最大得分的成员

    return predict_label


# 步骤五：性能评估
# 绘制混淆矩阵
#  cm 混淆矩阵，通常是一个二维数组，   classes 类别标签列表，用于在热力图中标记 x 和 y 轴，  normalize 布尔值，表示是否对混淆矩阵进行归一化。默认 F
def plot_confusion_matrix(cm, classes, normalize=False,
                          title='GMM-HMM 混淆矩阵',
                          cmap=plt.cm.Reds):    # cmap 配色表
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.show()


# 打印性能评估结果
def report(y_test, y_pred, show_cm=True):
    print("混淆矩阵:\n\n", confusion_matrix(y_test, y_pred))
    print("——————————————————————————————————————————————————————")
    print("——————————————————————————————————————————————————————\n")

    # classification_report函数
    # sklearn.metrics 模块中的一个函数，用于生成分类报告，显示主要的分类指标。
    # 精确度(查准率)：分类器正确预测为正类的样本数占所有预测为正类的样本数的比例。TP/(TP+FP)
    # 召回率(查全率)：分类器正确预测为正类的样本数占所有实际为正类的样本数的比例。TP/(TP+FN)
    # F1分数，精确度和召回率的调和平均值，用于平衡精确度和召回率。
    # 支持度：每个类别的样本数量
    print("分类结果:\n\n", classification_report(y_test, y_pred))
    print("——————————————————————————————————————————————————————")
    print("——————————————————————————————————————————————————————\n")

    # 准确率即混淆矩阵的正确性：(TP+TN)/(TP+TN+FP+FN)
    print("准确率:", accuracy_score(y_test, y_pred))
    print("——————————————————————————————————————————————————————")
    print("——————————————————————————————————————————————————————\n")

    if show_cm:
        plot_confusion_matrix(confusion_matrix(y_test, y_pred),
                              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


# DOWN-JSON
def save_to_json(data, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"数据已保存到 {file_name}")


if __name__ == '__main__':
    start_time = time.time()

    sound_path = r"/Users/dususu/Desktop/data/spoken_digit"
    x_train, y_train, x_test, y_test, data = build_dataset(sound_path)

    # print("训练集大小:", len(x_train))
    # print("测试集大小:", len(x_test))

    learned_hmm = train_model(data)

    # HMM 原始数据结构
    # print(learned_hmm)

    # HMM 参数保存为json
    # save_hmm_models_to_json(learned_hmm, "learned_hmm.json")

    with open("learned_hmm.pkl", "wb") as file:
        pickle.dump(learned_hmm, file)

    # 测试
    with open("learned_hmm.pkl", "rb") as file:
        # 从文件加载训练好的模型
        trained_model = pickle.load(file)

    y_pred = prediction(x_test, learned_hmm)
    print(y_pred)
    print("预测准确率：", accuracy_score(y_test, y_pred))

    report(y_test, y_pred, show_cm=True)

    # save_to_json({"learned_hmm": learned_hmm}, "learned_hmm.json")
    # save_to_json({"data": data}, "data.json")
    # save_to_json({"x_train": x_train, "y_train": y_train}, "train_data.json")
    # save_to_json({"x_test": x_test, "y_test": y_test}, "test_data.json")

    end_time = time.time()
    run_time = end_time - start_time
    print(f"代码运行时间: {run_time:.2f} 秒")