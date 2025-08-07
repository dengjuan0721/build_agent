```markdown
# Applications

## 经典自然语言处理应用

### 词性标注（POS Tagging）
作为HMM最成功的早期应用之一，1992年Church提出的Brill标注器使用HMM实现了90%+准确率。其核心是求解：
```python
import numpy as np
from hmmlearn import hmm

states = ["NN", "VB", "JJ"]  # 隐藏状态：词性标签
observations = ["word1", "word2"]  # 观测序列：单词

model = hmm.MultinomialHMM(n_components=len(states))
model.startprob_ = np.array([0.6, 0.3, 0.1])  # 初始概率
model.transmat_ = np.array([[0.7, 0.2, 0.1],  # 转移矩阵
                           [0.3, 0.5, 0.2],
                           [0.1, 0.4, 0.5]])
model.emissionprob_ = np.array([[0.8, 0.1],   # 发射矩阵
                               [0.1, 0.7],
                               [0.2, 0.3]])
```

### 命名实体识别（NER）
在1995年BBN的IDENTIFLEX系统中，HMM通过状态设计识别实体边界：
```python
states = ["O", "B-PER", "I-PER"]  # 非实体/人名开始/人名延续
transitions = {
    "O": {"O": 0.8, "B-PER": 0.2},
    "B-PER": {"I-PER": 0.7, "O": 0.3}
}
```

## 语音识别革命

### 声学模型建模
1980年代CMU的Sphinx系统采用：
```python
# 每个音素对应一个HMM状态
phoneme_hmm = {
    "AA": {"mean": 500, "var": 50},
    "IH": {"mean": 300, "var": 30}
}

# 使用GMM-HMM处理连续特征
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3)
gmm.fit(audio_features)  # MFCC特征
```

## 与其他序列模型对比

| 模型 | 训练方式 | 处理长距离依赖 | 典型准确率(NER) |
|-------|---------|---------------|----------------|
| HMM   | 监督学习 | 差            | 85%            |
| CRF   | 判别式  | 中等          | 89%            | 
| BiLSTM| 端到端  | 强            | 92%            |

```python
# CRF对比示例（使用sklearn-crfsuite）
import sklearn_crfsuite
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1
)
crf.fit(X_train, y_train)
```

## 局限性讨论

1. **强独立性假设**：观测独立性限制特征工程
```python
# 无法直接建模特征组合
hmm.emissionprob_ = [
    [P("word1|NN"), P("word2|NN")],  # 仅能处理单独特征
    ...
]
```

2. **上下文遗忘问题**：
```python
# 一阶马尔可夫性导致历史信息丢失
P(q_t|q_{t-1})  # 无法访问q_{t-2}
```

## 结论

通过词性标注、语音识别等经典案例，我们验证了HMM作为统计语言模型基石的价值。然而在需要复杂特征交互和长距离依赖的场景中，其理论限制逐渐显现。这为后续章节讨论的HMM与神经网络的融合埋下伏笔。
```