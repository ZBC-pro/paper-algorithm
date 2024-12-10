import os
import numpy as np
from scipy.io.wavfile import read
from sklearn.cluster import KMeans

# 预处理：对语音信号进行预加权，提升高频分量
def pre_emphasis(signal, pre_emphasis_coeff=0.97):
    """
    对语音信号进行预加权。
    :param signal: 原始语音信号
    :param pre_emphasis_coeff: 预加权系数，通常为0.97
    :return: 预加权后的信号
    """
    return np.append(signal[0], signal[1:] - pre_emphasis_coeff * signal[:-1])

# 分帧和加窗：对语音信号进行短时分帧并加Hamming窗
def framing_and_windowing(signal, sample_rate, frame_size=0.025, frame_stride=0.01):
    """
    对语音信号进行分帧和加窗处理。
    :param signal: 输入语音信号
    :param sample_rate: 语音采样率
    :param frame_size: 每帧的时间长度（秒）
    :param frame_stride: 帧移的时间长度（秒）
    :return: 分帧后的信号
    """
    # 计算帧长和帧移对应的采样点数
    frame_length = int(frame_size * sample_rate)
    frame_step = int(frame_stride * sample_rate)
    signal_length = len(signal)

    # 计算需要的帧数
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1

    # 填充零以确保最后一帧完整
    pad_signal_length = num_frames * frame_step + frame_length
    pad_signal = np.append(signal, np.zeros((pad_signal_length - signal_length)))

    # 提取帧的索引
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T

    # 分帧并加Hamming窗
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames *= np.hamming(frame_length)  # 应用Hamming窗
    return frames

# 自相关分析：计算帧内自相关值
def autocorrelation(frames):
    """
    对每一帧计算自相关值。
    :param frames: 分帧后的信号
    :return: 每帧的自相关结果
    """
    return np.array([np.correlate(frame, frame, mode='full')[len(frame) - 1:] for frame in frames])

# LPC倒谱系数提取：通过Levinson-Durbin算法求解
def lpc_analysis(frames, order=12):
    """
    提取每帧的LPC倒谱系数。
    :param frames: 分帧后的信号
    :param order: LPC阶数
    :return: 每帧的LPC倒谱系数
    """
    lpcs = []
    for frame in frames:
        # 计算自相关函数
        autocorr = np.correlate(frame, frame, mode='full')[len(frame) - 1:]
        R = autocorr[:order + 1]
        # 使用Levinson-Durbin算法求解LPC系数
        a = np.zeros(order + 1)
        e = np.zeros(order + 1)
        a[0] = 1
        e[0] = R[0]
        for i in range(1, order + 1):
            k = (R[i] - np.dot(a[:i], R[i - 1::-1])) / e[i - 1]
            a[i] = k
            a[:i] -= k * a[i - 1::-1]
            e[i] = (1 - k * k) * e[i - 1]
        lpcs.append(a[1:])  # 忽略第一个系数
    return np.array(lpcs)

# 动态特征计算：计算每帧特征的差分特征
def calculate_delta(features, N=2):
    """
    计算特征的动态变化（差分特征）。
    :param features: 输入特征矩阵 (帧数 x 特征维度)
    :param N: 差分窗口大小
    :return: 差分特征矩阵
    """
    num_frames = features.shape[0]
    denominator = 2 * sum([i ** 2 for i in range(1, N + 1)])
    delta_feat = np.zeros_like(features)

    # 在边缘填充以确保边界特征完整
    padded = np.pad(features, ((N, N), (0, 0)), mode='edge')
    for t in range(num_frames):
        delta_feat[t] = np.sum([(n * (padded[t + n + N] - padded[t - n + N]))
                                for n in range(1, N + 1)], axis=0) / denominator
    return delta_feat

# 矢量量化：对特征进行k-means聚类
# 修改 vector_quantization 函数以返回分配的类别及其聚类中心
def vector_quantization(features, n_clusters=32):
    """
    使用k-means对特征进行矢量量化，生成码本。
    :param features: 输入特征
    :param n_clusters: 码本大小（聚类数）
    :return: 聚类模型和每个特征对应的码字索引
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(features)
    return kmeans, kmeans.predict(features)

# 遍历音频目录并提取特征
def process_directory(directory, n_clusters=32):
    """
    遍历目录中的音频文件，提取特征并生成码本。
    :param directory: 音频文件所在目录
    :param n_clusters: 聚类数（码本大小）
    :return: 提取的特征和标签
    """
    all_features = []
    labels = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".wav"):
            # 从文件名提取标签（数字部分）
            label = int(file_name.split('_')[0])
            sample_rate, signal = read(os.path.join(directory, file_name))
            signal = pre_emphasis(signal)
            print(signal)
            frames = framing_and_windowing(signal, sample_rate)
            lpc_features = lpc_analysis(frames)
            delta_features = calculate_delta(lpc_features)
            combined_features = np.hstack((lpc_features, delta_features))
            all_features.append(combined_features)
            labels.append(label)

    # 矢量量化并生成码本
    flat_features = np.vstack(all_features)
    kmeans_model, quantized_features = vector_quantization(flat_features, n_clusters)
    training_data = [(quantized_features[i], labels[i]) for i in range(len(labels))]

    return training_data, kmeans_model

if __name__ == "__main__":
    data_dir = r"/Users/dususu/Desktop/data/70/train"  # 替换为数据集路径
    training_data, kmeans_model = process_directory(data_dir)

    # 打印 K-Means 聚类结果
    # print("K-Means Cluster Centers:")
    # print(kmeans_model.cluster_centers_)  # 打印聚类中心

    # print("\nCluster Assignment:")
    # 打印每个数据点的分类结果
    # for i, label in enumerate(kmeans_model.labels_):
        # print(f"Sample {i} assigned to Cluster {label}")