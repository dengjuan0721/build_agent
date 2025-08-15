### 2. RNN基础：理解循环神经网络 (RNN Basics: Understanding Recurrent Networks)  

在传统神经网络中，我们习惯于处理固定大小的输入——比如一张图片或一个特征向量。但当我们面对文本、语音或时间序列这类**序列数据**时，固定输入的结构就显得力不从心了。这就是循环神经网络(RNN)大显身手的地方。  

#### 固定输入VS序列输入的对比  
想象你在阅读一篇文章：  
- **传统网络**：必须一次性接收整篇文章作为输入，且无法理解单词之间的顺序关系  
- **RNN**：像人类一样逐字阅读，每个时间步处理一个字符/单词，同时记住之前阅读过的内容  

这种差异就像比较拍照和看电影——前者是静态快照，后者是动态理解情节发展。  

为了更直观地感受这种差异，下面用几行代码演示两种处理方式：

```python
import numpy as np

# 模拟一个简单句子（5个词，每个词用4维向量表示）
sentence = np.random.randn(5, 4)  # shape: (seq_len, input_dim)

# 1. 传统全连接网络：只能接受固定长度输入
def traditional_forward(x_flat):
    # 把序列展平成一个长向量
    x_flat = x_flat.reshape(-1)  # 20维
    W = np.random.randn(20, 10)  # 固定权重
    return np.tanh(W.T @ x_flat)

fixed_output = traditional_forward(sentence)
print("传统网络输出形状:", fixed_output.shape)  # (10,)

# 2. RNN：逐时间步处理，保留状态
def rnn_step(x_t, h_prev, W_xh, W_hh):
    """单步RNN前向"""
    h_t = np.tanh(W_xh @ x_t + W_hh @ h_prev)
    return h_t

# 初始化参数
W_xh = np.random.randn(8, 4)   # 输入到隐藏
W_hh = np.random.randn(8, 8)   # 隐藏到隐藏
h = np.zeros(8)                # 初始隐藏状态

# 逐词处理
for t, word_vec in enumerate(sentence):
    h = rnn_step(word_vec, h, W_xh, W_hh)
    print(f"第{t+1}个词处理后隐藏状态: {h[:3]}...")  # 只看前3维
```

#### RNN的核心API：step函数  
每个RNN单元的核心是一个简单的`step`函数，可以理解为网络的"记忆处理器"：  
```python
def step(x, h_prev):
    h_new = tanh(dot(W_hh, h_prev) + dot(W_xh, x))
    y = dot(W_hy, h_new)
    return h_new, y
```  
这个函数完成三个关键操作：  
1. 将当前输入x与前一刻隐藏状态h_prev结合  
2. 通过tanh激活函数生成新状态  
3. 输出预测结果y  

下面给出一个可运行的极简RNN单元实现，并展示它如何在一个字符级任务上工作：

```python
import numpy as np

class MinimalRNNCell:
    """一个极简的RNN单元，仅作教学演示"""
    def __init__(self, input_size, hidden_size, output_size):
        # 随机初始化权重
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.1
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.1
        self.b_h  = np.zeros(hidden_size)
        self.b_y  = np.zeros(output_size)

    def step(self, x, h_prev):
        """
        单步前向
        x: 当前输入 (input_size,)
        h_prev: 上一时刻隐藏状态 (hidden_size,)
        返回 (h_new, y)
        """
        h_new = np.tanh(self.W_xh @ x + self.W_hh @ h_prev + self.b_h)
        y = self.W_hy @ h_new + self.b_y
        return h_new, y

# 示例：字符级RNN预测下一个字符
chars = ['h', 'i', ' ']
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for c, i in char2idx.items()}

# 把字符转成one-hot向量
def char_to_onehot(c):
    vec = np.zeros(len(chars))
    vec[char2idx[c]] = 1
    return vec

rnn = MinimalRNNCell(input_size=3, hidden_size=5, output_size=3)

# 输入序列 "hi "，预测下一个字符
seq = "hi "
h = np.zeros(5)  # 初始隐藏状态
for c in seq:
    x = char_to_onehot(c)
    h, logits = rnn.step(x, h)
    probs = np.exp(logits) / np.sum(np.exp(logits))  # softmax
    next_char = idx2char[np.argmax(probs)]
    print(f"输入'{c}' -> 预测下一个字符概率: {dict(zip(chars, np.round(probs, 2)))}")
```

#### 隐藏状态的数学本质  
那个神秘的h状态可以用这个方程表示：  $h_t = tanh(W_hh h_{t-1} + W_xh x_t)$

让我们拆解这个公式：  
- $W_hh$：控制如何保留历史信息（记忆权重）  
- $W_xh$：决定当前输入的重要性（输入权重）  
- $tanh$：非线性激活，将值压缩到(-1,1)范围  

这就像你在读书时：新理解($h_t$)是当前段落($x_t$)和你之前的知识($h_{t-1}$)的综合体。  

下面用代码一步步展示这个公式如何实际计算：

```python
# 手动计算隐藏状态，帮助理解公式
seq_len = 3
hidden_size = 4
input_dim  = 2

# 随机参数
W_hh = np.random.randn(hidden_size, hidden_size)
W_xh = np.random.randn(hidden_size, input_dim)
h = np.zeros(hidden_size)  # h_{t-1}

# 模拟输入序列
x_series = [np.random.randn(input_dim) for _ in range(seq_len)]

print("手动计算隐藏状态：")
for t, x_t in enumerate(x_series, 1):
    # 严格按照公式计算
    h = np.tanh(W_hh @ h + W_xh @ x_t)
    print(f"t={t}, h_t = {np.round(h, 3)}")
```

#### 深度RNN：堆叠的智慧  
当简单RNN不够用时，我们可以堆叠多层网络：  
```
输入 → RNN层1 → RNN层2 → ... → RNN层N → 输出  
```  
每层都维护自己的隐藏状态，底层捕捉局部模式（如字母组合），高层学习全局结构（如语法规则）。实验显示，3-5层的深度RNN在语言建模中表现最佳。  

下面用PyTorch快速搭建一个2层RNN，并展示各层隐藏状态的维度：

```python
import torch
import torch.nn as nn

# 构造一个简单的2层RNN
vocab_size = 50
embed_dim  = 16
hidden_dim = 32
num_layers = 2

rnn = nn.RNN(input_size=embed_dim,
             hidden_size=hidden_dim,
             num_layers=num_layers,
             batch_first=True)  # 输入形状 (batch, seq, feature)

# 假设一个batch有3条序列，每条长7
batch_size = 3
seq_len = 7
x = torch.randn(batch_size, seq_len, embed_dim)

# 前向传播
out, h_n = rnn(x)  # out: (batch, seq, hidden), h_n: (layers, batch, hidden)

print("输出形状:", out.shape)        # (3, 7, 32)
print("最后一层隐藏状态形状:", h_n.shape)  # (2, 3, 32)

# 查看各层隐藏状态
for layer_idx in range(num_layers):
    print(f"第{layer_idx+1}层最后一步隐藏状态: {h_n[layer_idx, 0, :5]}...")  # 只看前5维
```
RNN的这种时序处理能力虽然现在被Transformer部分取代，但它仍然是理解序列建模思想的绝佳起点。当你看到那个简单的step函数如何通过循环调用处理任意长度输入时，会感受到神经网络设计的美学——用最小化的接口实现最大化的表达能力。