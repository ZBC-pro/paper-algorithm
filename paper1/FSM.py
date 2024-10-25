'''
实现思路：将想要获取的事件序列拆分为多个wait()列表
'''

# 创建多列表集合
from collections import defaultdict

#定义选片结构
class Episode:
    def __init__(self, sequence):
        self.sequence = sequence    # 定义候选片序列，如[B, C]
        self.freq = 0   # 初始化频率

# 频繁片段检测算法
def frequent_episodes(candidate_episodes, event_stream, lambda_min):
    print(candidate_episodes)
    waits = defaultdict(list)   # 存储等待每个事件的自动机状态
    bag = []    # 临时存储自动机状态

    # 初始化waits列表    dict.fromkeys去重并且保留原始顺序
    for candidate_event in list(dict.fromkeys([e[0] for e in event_stream])):
        print(candidate_event)
        waits[candidate_event] = []

    # 初始化所有的候选片段
    for episode in candidate_episodes:
        waits[episode.sequence[0]].append((episode, 1))     # 初始化每个候选片段等待的第一个事件
        episode.freq = 0

    # 遍历事件流
    for event, timestamp in event_stream:
        current_wait_list = waits[event][:]     # 获取当前等待事件的自动机列表，使用副本避免修改


if __name__ == "__main__":
    #  需要输出的候选片段
    candidate_episodes = [Episode(['B', 'C'])]

    # 事件序列
    event_stream = [('A', 1), ('B', 3), ('D', 4), ('C', 6), ('E', 12), ('A', 14), ('B', 15), ('C', 17)]

    lambda_min = 0.2

    frequent_episodes(candidate_episodes, event_stream, lambda_min)
