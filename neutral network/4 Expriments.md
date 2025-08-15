### 4. 趣味实验：不同领域的RNN表现 (Fun Experiments: RNN Across Domains)  

当RNN掌握了字符级语言建模的基本能力后，最令人兴奋的部分莫过于观察它在不同领域的生成表现。这些实验不仅展示了RNN的通用性，更揭示了神经网络如何捕捉不同文本风格的潜在规律。  

#### 保罗·格雷厄姆散文风格生成  
训练RNN阅读数十篇保罗·格雷厄姆（Paul Graham）的科技散文后，模型开始生成极具辨识度的文本："创业公司的核心优势在于它能以指数级速度学习，而大公司只能线性进步..."。有趣的是，模型不仅学会了格雷厄姆标志性的论点递进结构，还模仿了他常用的分号使用习惯。这证明RNN确实能捕捉作者特有的句法模式和思维节奏。

下面是一个简化的PyTorch实现，展示如何为保罗·格雷厄姆散文训练一个字符级RNN：

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 1. 准备数据集
class PGDataset(Dataset):
    def __init__(self, text_file, seq_length=100):
        with open(text_file, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        # 创建字符到索引的映射
        self.chars = sorted(list(set(self.text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        
        self.seq_length = seq_length
        self.vocab_size = len(self.chars)
        
    def __len__(self):
        return len(self.text) - self.seq_length
    
    def __getitem__(self, idx):
        chunk = self.text[idx:idx+self.seq_length+1]
        input_seq = [self.char_to_idx[ch] for ch in chunk[:-1]]
        target_seq = [self.char_to_idx[ch] for ch in chunk[1:]]
        return torch.tensor(input_seq), torch.tensor(target_seq)

# 2. 定义RNN模型
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, state=None):
        x = self.embed(x)  # (batch, seq_len, embed_size)
        output, state = self.rnn(x, state)
        output = self.fc(output)  # (batch, seq_len, vocab_size)
        return output, state

# 3. 训练循环示例
def train_model(model, dataloader, num_epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
            # 生成示例文本
            print(generate_text(model, dataloader.dataset, "创业公司的核心", 200))

# 4. 文本生成函数
def generate_text(model, dataset, seed_text, length=200, temperature=0.8):
    model.eval()
    device = next(model.parameters()).device
    
    # 将种子文本转换为索引
    input_seq = torch.tensor([[dataset.char_to_idx[ch] for ch in seed_text]]).to(device)
    
    generated = seed_text
    hidden = None
    
    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input_seq, hidden)
            
            # 使用温度采样
            logits = output[0, -1] / temperature
            probs = torch.softmax(logits, dim=0)
            next_char_idx = torch.multinomial(probs, 1).item()
            
            next_char = dataset.idx_to_char[next_char_idx]
            generated += next_char
            
            # 更新输入
            input_seq = torch.tensor([[next_char_idx]]).to(device)
    
    return generated
```

#### 莎士比亚戏剧创作  
用莎士比亚全集训练的RNN会生成令人捧腹又惊叹的"伪莎剧"："SCENE V. Elsinore. A room in the castle. Enter HAMLET and POLONIUS POLONIUS: My lord, the queen would speak with you. HAMLET: O God, I could be bounded in a nutshell..." 尽管角色对话有时会逻辑断裂，但抑扬格五音步(iambic pentameter)的韵律和古英语词汇的使用令人称奇。实验显示，模型约在训练20轮后突然"顿悟"了引号配对和换行格式。

为了验证模型对莎士比亚风格的掌握程度，我们可以计算生成文本的韵律特征：

```python
import re
from collections import Counter

def analyze_shakespeare_style(text):
    """分析文本的莎士比亚风格特征"""
    
    # 1. 检测抑扬格五音步（10个音节一行）
    lines = text.split('\n')
    pentameter_lines = 0
    
    # 简化的音节计数（实际应用中需要更复杂的算法）
    for line in lines:
        # 移除标点符号
        clean_line = re.sub(r'[^\w\s]', '', line.lower())
        words = clean_line.split()
        
        # 估算音节数（基于元音组）
        syllable_count = 0
        for word in words:
            vowels = re.findall(r'[aeiouy]+', word)
            syllable_count += max(1, len(vowels))
        
        if 8 <= syllable_count <= 12:  # 允许一定误差
            pentameter_lines += 1
    
    # 2. 统计古英语词汇使用
    archaic_words = ['thee', 'thou', 'thy', 'hath', 'doth', 'art', 'hath', 'ere']
    archaic_count = sum(text.lower().count(word) for word in archaic_words)
    
    # 3. 检测戏剧格式
    scene_pattern = r'SCENE\s+[IVXLCDM]+\.'
    scenes = re.findall(scene_pattern, text)
    
    return {
        'pentameter_ratio': pentameter_lines / max(len(lines), 1),
        'archaic_word_density': archaic_count / max(len(text.split()), 1),
        'scene_count': len(scenes)
    }

# 使用示例
generated_shakespeare = """
SCENE II. A room in the castle.
Enter HAMLET and HORATIO
HAMLET: Good morrow, friend! What news dost thou bring?
HORATIO: My lord, the king is most displeased with thee.
"""

metrics = analyze_shakespeare_style(generated_shakespeare)
print(f"五音步比例: {metrics['pentameter_ratio']:.2f}")
print(f"古英语密度: {metrics['archaic_word_density']:.4f}")
print(f"场景数量: {metrics['scene_count']}")
```
