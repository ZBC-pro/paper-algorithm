from collections import defaultdict

class Episode:
    def __init__(self, sequence):
        self.sequence = sequence  # 定义所有候选序列，如A-B-C，当状态完成这三个时，频率+1
        self.freq = 0

    def __repr__(self):
        return f"Episode(sequence={self.sequence}, freq={self.freq})"

# 初始化候选片段的等待状态
def init_waits(candidate_episodes):
    waits = defaultdict(list)
    for idx, episode in enumerate(candidate_episodes, start=1):
        first_event = episode.sequence[0]
        waits[first_event].append((episode, 0))
        print(f"候选片段{idx} 初始化：waits({first_event}) = {(episode, 1)}")
    return waits

# 移动事件到下一个并改变状态
def remove_events(event_stream, waits):
    for event, timestamp in event_stream:
        if event in waits:
            current_wait_list = waits[event][:]  # 使用副本以避免直接修改
            temp_bag = []  # 使用temp_bag暂存更新状态

            for episode, state in current_wait_list:
                next_state = state + 1
                if next_state >= len(episode.sequence):
                    # 匹配完成，增加频率，并重置状态
                    episode.freq += 1
                    print(f"{episode.sequence} 匹配完成，增加频率：{episode.freq}")
                    first_event = episode.sequence[0]
                    temp_bag.append((episode, 0))  # 重置状态为 0
                else:
                    next_event = episode.sequence[next_state]
                    temp_bag.append((episode, next_state))

                waits[event].remove((episode, state))  # 从当前等待列表中移除旧状态

            # 将 temp_bag 中的状态更新到 waits 中，批量更新
            for episode, next_state in temp_bag:
                waits[episode.sequence[next_state]].append((episode, next_state))

    print("\n最终的候选片段频率：")
    for episode in candidate_episodes:
        print(f"{episode.sequence}: 出现次数 {episode.freq}")

# 测试代码
if __name__ == "__main__":
    candidate_episodes = [Episode(['A', 'B', 'C']), Episode(['B', 'C'])]
    event_stream = [('A', 1), ('B', 3), ('D', 4), ('C', 6), ('E', 12), ('A', 14), ('B', 15), ('C', 17), ('B', 18),
                    ('C', 20)]

    waits = init_waits(candidate_episodes)

    # 初始化后的 waits 列表
    print("\n初始化后的 waits 状态：")
    for event, waiting_list in waits.items():
        print(f"waits({event}) = {waiting_list}")

    remove_events(event_stream, waits)