```markdown
# Introduction

隐马尔可夫模型（Hidden Markov Model, HMM）是一种经典的统计语言模型，用于描述由隐藏状态序列生成观测序列的概率过程。其核心思想是假设系统状态不可直接观测（隐藏），但能通过观测到的数据间接推断状态变化规律。

## HMM的基本概念

HMM通过双重随机过程建模：
1. **状态序列**：系统内部隐藏的状态转移过程（如天气变化）
2. **观测序列**：每个状态生成的可见观测结果（如观察到的行为）

```python
# 伪代码示例：HMM生成观测序列
hidden_states = ["晴天", "雨天"]
observations = ["散步", "宅家"]

current_state = random.choice(hidden_states)
for _ in range(10):
    emit_observation = random.choice(observations if current_state == "晴天" else reversed(observations))
    print(f"状态:{current_state} -> 观测:{emit_observation}")
    current_state = random.choice(hidden_states)  # 状态转移
```

## 核心组成部分

1. **状态集合**：系统的可能隐藏状态（如{Noun, Verb}词性标签）
2. **观测集合**：实际可见的输出（如单词序列）
3. **转移概率**：状态间的转换规律（如"Noun后接Verb的概率"）
4. **发射概率**：某状态下生成特定观测的概率（如"Noun状态下出现'苹果'的概率"）

```python
# 概率表示示例（非真实值）
transition_prob = {
    "晴天": {"晴天": 0.7, "雨天": 0.3},
    "雨天": {"晴天": 0.4, "雨天": 0.6}
}

emission_prob = {
    "晴天": {"散步": 0.8, "宅家": 0.2},
    "雨天": {"散步": 0.1, "宅家": 0.9}
}
```

## 典型应用场景

1. **自然语言处理**：
   - 词性标注（通过单词序列推断词性状态）
   - 语音识别（声学信号→文字）

2. **生物信息学**：
   - DNA序列分析
   - 蛋白质结构预测

3. **时序数据分析**：
   - 股票市场趋势预测
   - 手势识别

```python
# 应用示例：词性标注
states = ["N", "V"]
observations = ["fly", "apple"]

# Viterbi算法伪代码
def decode(obs_sequence):
    # 动态规划计算最优状态路径
    return best_path
```

## 本章小结

本节介绍了HMM的基本概念和核心组成部分，为后续深入探讨其算法和应用奠定基础。接下来将详细解析HMM的三个经典问题：评估问题、解码问题和学习问题。
```