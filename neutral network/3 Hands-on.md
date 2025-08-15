### 3. 实战：字符级语言模型 (Hands-on: Character-Level Language Models)

字符级语言模型是RNN最直观的应用之一——它像婴儿学语一样，从零开始学习文本的统计规律。让我们拆解这个神奇过程的实现细节：

#### 训练数据准备：文本文件的处理
1. **原始文本处理**：任何纯文本文件（.txt）都可作为训练数据。建议从1MB左右的小规模文本开始（如莎士比亚全集），便于快速迭代
2. **字符级处理**：与单词级不同，我们直接处理原始ASCII/Unicode字符。例如句子"The"会被拆解为['T', 'h', 'e']
3. **字符表构建**：统计所有出现过的字符构建词汇表。英文通常有50-100个字符（大小写字母+标点），中文则需要处理数千个汉字

```python
# 1. 读取并预处理文本
with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 2. 构建字符到索引的映射
chars = sorted(list(set(text)))  # 获取所有唯一字符
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# 3. 将文本转换为字符索引序列
text_as_int = [char_to_idx[c] for c in text]

print(f"文本长度: {len(text)} 字符")
print(f"词汇表大小: {len(chars)}")
print(f"前20个字符: {text[:20]}")
print(f"对应的索引: {text_as_int[:20]}")
```

#### 1-of-k编码的字符表示
每个字符被编码为稀疏向量：
- 词汇表大小为k（如80个字符）
- 字符"a"表示为长度为80的向量，仅在对应索引位置为1，其余为0
- 例如：若词汇表是['a','b','c']，则"b"编码为[0,1,0]

这种表示虽然简单，但能保留字符间的独立性。实践中我们会使用嵌入层(Embedding Layer)将其转换为密集向量表示。

```python
import numpy as np

def one_hot_encode(char_idx, vocab_size):
    """将字符索引转换为one-hot向量"""
    one_hot = np.zeros(vocab_size)
    one_hot[char_idx] = 1
    return one_hot

# 示例：编码字符'h'
vocab_size = len(chars)
char_h_idx = char_to_idx['h']
h_one_hot = one_hot_encode(char_h_idx, vocab_size)

print(f"字符 'h' 的索引: {char_h_idx}")
print(f"One-hot向量 (前10维): {h_one_hot[:10]}")
print(f"向量总和: {h_one_hot.sum()}")  # 应该为1
```

#### 损失函数：Softmax分类器的应用
在每个时间步t：
1. RNN输出维度为k的向量$o_t$
2. 通过Softmax转换为概率分布：
$p_t = softmax(o_t)$
3. 计算交叉熵损失：$L_t = -log(p_t[y_t])$，其中$y_t$是真实的下一个字符
4. 整个序列的损失是各时间步损失的平均值

这个设计使得模型学习最大化真实字符的预测概率。有趣的是，即使简单的RNN也能学习到字符间的长距离依赖——比如在开引号后更可能预测闭引号。

```python
import numpy as np

def softmax(logits):
    """将原始分数转换为概率分布"""
    exp_logits = np.exp(logits - np.max(logits))  # 数值稳定性技巧
    return exp_logits / np.sum(exp_logits)

def cross_entropy_loss(probs, target_idx):
    """计算交叉熵损失"""
    return -np.log(probs[target_idx] + 1e-8)  # 添加小值避免log(0)

# 示例：模拟RNN输出
vocab_size = len(chars)
rnn_output = np.random.randn(vocab_size)  # 模拟RNN输出
probs = softmax(rnn_output)

# 假设真实下一个字符是'e'
true_char = 'e'
target_idx = char_to_idx[true_char]
loss = cross_entropy_loss(probs, target_idx)

print(f"RNN输出维度: {rnn_output.shape}")
print(f"预测概率总和: {probs.sum():.4f}")  # 应该接近1.0
print(f"真实字符 '{true_char}' 的预测概率: {probs[target_idx]:.4f}")
print(f"交叉熵损失: {loss:.4f}")
```

#### 反向传播通过时间的实现细节
BPTT(Backpropagation Through Time)是训练RNN的核心算法：
1. **展开计算图**：将RNN按时间步展开成深度前馈网络
2. **梯度流动**：损失函数的梯度会沿着时间维度反向传播
3. **梯度消失问题**：原始RNN中，梯度会指数级衰减（这也是LSTM被提出的原因）
4. **截断BPTT**：对于长序列，通常只反向传播固定步数（如50步）以节省计算资源

关键代码结构示例（伪代码）：

```python
import numpy as np

class SimpleRNN:
    def __init__(self, vocab_size, hidden_size):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # 初始化权重矩阵
        self.Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(vocab_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))
        
    def forward_step(self, x, h_prev):
        """单步前向传播"""
        # x: one-hot输入 (vocab_size, 1)
        # h_prev: 上一时刻的隐藏状态 (hidden_size, 1)
        
        h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h_prev) + self.bh)
        y = np.dot(self.Why, h) + self.by
        p = softmax(y.flatten())
        
        return h, y, p
    
    def train_step(self, inputs, targets, hprev, learning_rate=1e-3):
        """截断BPTT的单次训练步骤"""
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0
        
        # 前向传播
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1  # one-hot编码
            
            hs[t], ys[t], ps[t] = self.forward_step(xs[t], hs[t-1])
            loss += cross_entropy_loss(ps[t], targets[t])
        
        # 反向传播
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])
        
        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1  # 交叉熵梯度
            
            dWhy += np.dot(dy.reshape(-1, 1), hs[t].T)
            dby += dy.reshape(-1, 1)
            
            dh = np.dot(self.Why.T, dy.reshape(-1, 1)) + dhnext
            dhraw = (1 - hs[t] * hs[t]) * dh  # tanh的导数
            
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(self.Whh.T, dhraw)
        
        # 梯度裁剪防止梯度爆炸
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
        
        # 更新参数
        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby
        
        return loss, hs[len(inputs)-1]

# 使用示例
vocab_size = len(chars)
hidden_size = 100
rnn = SimpleRNN(vocab_size, hidden_size)

# 准备训练序列
seq_length = 25
for i in range(0, len(text_as_int) - seq_length, seq_length):
    inputs = text_as_int[i:i+seq_length]
    targets = text_as_int[i+1:i+seq_length+1]
    
    # 训练一步
    loss, hprev = rnn.train_step(inputs, targets, np.zeros((hidden_size, 1)))
    
    if i % 10000 == 0:
        print(f"迭代 {i}, 损失: {loss:.4f}")
```

通过这种端到端的训练，RNN会逐渐从最初随机预测字符，发展到能生成符合语法和语义的文本——这个过程就像观察一个数字生命学习人类语言般奇妙。