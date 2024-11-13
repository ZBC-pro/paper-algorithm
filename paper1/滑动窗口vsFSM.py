import time
import random
import matplotlib.pyplot as plt
from collections import defaultdict

class Episode:
    def __init__(self, sequence):
        self.sequence = sequence    # 定义所有候选序列，如A-B-C，当状态完成这三个时，频率+1
        self.freq = 0

    def __repr__(self):
        return f"Episode(sequence={self.sequence}, freq={self.freq})"

# 生成带噪声的事件流
def generate_noisy_event_stream(length, noise_rate=0.7):
    event_stream = [('A', i) if i % 5 == 0 else ('B', i) if i % 5 == 1 else ('C', i) if i % 5 == 2 else ('D', i) if i % 5 == 3 else ('E', i) for i in range(length)]
    noise_events = ['F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
    noisy_event_stream = []

    for event in event_stream:
        noisy_event_stream.append(event)
        if random.random() < noise_rate:
            noise_event = random.choice(noise_events)
            timestamp = event[1] + 0.5
            noisy_event_stream.append((noise_event, timestamp))

    return noisy_event_stream

# 滑动窗口方法
def sliding_window_count(event_stream, candidate_episodes):
    frequencies = {str(episode.sequence): 0 for episode in candidate_episodes}
    max_length = max(len(ep.sequence) for ep in candidate_episodes)

    for i in range(len(event_stream) - max_length + 1):
        window = [event for event, _ in event_stream[i:i + max_length]]
        for episode in candidate_episodes:
            if window[:len(episode.sequence)] == episode.sequence:
                frequencies[str(episode.sequence)] += 1

    return frequencies

# 初始化候选片段的等待状态
def init_waits(candidate_episodes):
    waits = defaultdict(list)
    for episode in candidate_episodes:
        first_event = episode.sequence[0]
        waits[first_event].append((episode, 0))
    return waits

# EGH方法
def remove_events(event_stream, waits):
    for event, timestamp in event_stream:
        if event in waits:
            current_wait_list = waits[event][:]
            for episode, state in current_wait_list:
                next_state = state + 1
                if next_state >= len(episode.sequence):
                    episode.freq += 1
                    first_event = episode.sequence[0]
                    waits[first_event].append((episode, 0))
                else:
                    next_event = episode.sequence[next_state]
                    waits[next_event].append((episode, next_state))
                waits[event].remove((episode, state))

# 对比不同噪声水平下的时间消耗
def compare_time_by_noise_levels(candidate_episodes, length=20000, noise_levels=[0.1, 0.5, 0.9], step=2000):
    sizes = list(range(step, length + 1, step))

    for noise_rate in noise_levels:
        # 生成带噪声的事件流
        noisy_event_stream = generate_noisy_event_stream(length, noise_rate)

        sliding_times = []
        egh_times = []

        for size in sizes:
            subset_stream = noisy_event_stream[:size]

            # 滑动窗口方法
            start_time = time.time()
            sliding_window_count(subset_stream, candidate_episodes)
            sliding_times.append(time.time() - start_time)

            # EGH方法
            waits = init_waits(candidate_episodes)
            start_time = time.time()
            remove_events(subset_stream, waits)
            egh_times.append(time.time() - start_time)

        # 为当前噪声水平绘制时间对比图
        plt.figure()
        plt.plot(sizes, sliding_times, label="Sliding Window Method")
        plt.plot(sizes, egh_times, label="EGH Method")
        plt.xlabel("Event Stream Size")
        plt.ylabel("Execution Time (seconds)")
        plt.title(f"Execution Time Comparison (Noise={noise_rate})")
        plt.legend()
        plt.show()

# 示例运行
if __name__ == "__main__":
    # 候选片段包含频繁模式
    candidate_episodes = [
        Episode(['A', 'B', 'C']),
        Episode(['B', 'C', 'D']),
        Episode(['C', 'D', 'E']),
        Episode(['A', 'D', 'C']),
        Episode(['B', 'E'])
    ]

    # 对比不同噪声水平下的滑动窗口方法和EGH方法
    compare_time_by_noise_levels(candidate_episodes, length=20000, noise_levels=[0.1, 0.5, 0.9], step=2000)