import os
import librosa
import numpy as np
from hmmlearn.hmm import GaussianHMM
import librosa
import glob

# 1. 音频加载和特征提取
def extract_features(audio_path, n_mfcc=13):
    """
    提取音频的 MFCC 特征。
    :param audio_path: 音频文件路径。
    :param n_mfcc: 提取的 MFCC 系数数目。
    :return: MFCC 特征矩阵。
    """
    y, sr = librosa.load(audio_path, sr=None)
    # 动态调整 n_fft，确保它不超过信号长度
    n_fft = min(2048, len(y))
    hop_length = n_fft // 2  # 通常 hop_length 是 n_fft 的一半

    # 使用动态调整的 n_fft 和 hop_length
    mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfcc_features.T  # 每行是一个帧的特征


# 2. 加载数据
def load_data(data_dir):
    """
    加载数据集并提取特征。
    :param data_dir: 数据集根目录，按类别存储文件夹 (e.g., 0, 1, ..., 9)。
    :return: 特征列表和标签列表。
    """
    features = []
    labels = []
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                if file.endswith('.wav'):
                    file_path = os.path.join(label_path, file)
                    mfcc = extract_features(file_path)
                    features.append(mfcc)
                    labels.append(label)
    return features, labels


# 3. 模型训练
def train_gmm_hmm_models(features, labels, n_states, n_mixtures):
    """
    为每个类别训练一个 GMM-HMM 模型。
    :param features: 特征列表。
    :param labels: 标签列表。
    :param n_states: HMM 的隐藏状态数量。
    :param n_mixtures: 每个状态的 GMM 混合成分数量。
    :return: 每个类别的 GMM-HMM 模型字典。
    """
    unique_labels = sorted(set(labels))
    models = {}

    for label in unique_labels:
        # 获取当前类别的所有特征
        label_features = [features[i] for i in range(len(labels)) if labels[i] == label]
        lengths = [len(f) for f in label_features]
        concatenated_features = np.vstack(label_features)

        # 创建 GMM-HMM 模型
        model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100, tol=1e-4, random_state=42)
        model.fit(concatenated_features, lengths=lengths)  # 训练模型
        models[label] = model

    return models


# 4. 识别
def recognize(audio_path, models):
    """
    对输入音频进行识别。
    :param audio_path: 音频文件路径。
    :param models: 已训练的 GMM-HMM 模型字典。
    :return: 预测的类别。
    """
    mfcc = extract_features(audio_path)
    best_score = float('-inf')
    best_label = None

    for label, model in models.items():
        score = model.score(mfcc)
        if score > best_score:
            best_score = score
            best_label = label

    return best_label


# 主程序
if __name__ == "__main__":
    # 数据路径
    train_dir = r"/Users/dususu/Desktop/data/test/train"
    test_dir = r"/Users/dususu/Desktop/data/test/test"

    # 参数
    n_mfcc = 13  # MFCC 特征维度
    n_states = 5  # HMM 隐藏状态数量
    n_mixtures = 3  # 每个状态的 GMM 混合成分数量

    # 步骤 1: 加载训练数据
    print("加载训练数据...")
    train_features, train_labels = load_data(train_dir)

    # 步骤 2: 训练 GMM-HMM 模型
    print("训练 GMM-HMM 模型...")
    gmm_hmm_models = train_gmm_hmm_models(train_features, train_labels, n_states, n_mixtures)

    # 步骤 3: 加载测试数据并识别
    print("加载测试数据并进行识别...")
    test_features, test_labels = load_data(test_dir)
    correct = 0


    for i, feature in enumerate(test_features):
        # 使用 glob 匹配文件
        pattern = os.path.join(test_dir, test_labels[i], f"test_0_*.wav")  # 示例路径
        print(pattern)
        matching_files = glob.glob(pattern)

        if not matching_files:
            print(f"未找到匹配的测试文件: {pattern}")
            continue

        # 如果有多个匹配文件，选择第一个
        test_audio_path = matching_files[0]
        predicted_label = recognize(test_audio_path, gmm_hmm_models)
        print(f"音频: {test_audio_path}, 预测: {predicted_label}, 实际: {test_labels[i]}")

        if predicted_label == test_labels[i]:
            correct += 1

    # 打印准确率
    accuracy = correct / len(test_labels)
    print(f"识别准确率: {accuracy:.2%}")