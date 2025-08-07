```markdown
# Learning Problem

## 1. HMM学习问题的定义

隐马尔可夫模型(HMM)的学习问题可形式化为：给定观测序列$O=(o_1,...,o_T)$，估计模型参数$\lambda=(A,B,\pi)$使得$P(O|\lambda)$最大化。其中：
- $A$是状态转移矩阵
- $B$是观测概率矩阵
- $\pi$是初始状态分布

该问题本质上是参数估计问题，但由于存在隐变量（状态序列），无法直接使用最大似然估计。

```python
# 示例：HMM参数结构
import numpy as np

states = ['S1', 'S2']
observations = ['A', 'B']

# 初始化参数
A = np.array([[0.6, 0.4],  # 状态转移矩阵
              [0.3, 0.7]])  
B = np.array([[0.8, 0.2],  # 观测概率矩阵
              [0.5, 0.5]])
pi = np.array([0.7, 0.3])  # 初始分布
```

## 2. Baum-Welch算法的EM框架

Baum-Welch算法采用EM框架解决学习问题：

**E-Step**：计算期望统计量
- $\gamma_t(i)=P(q_t=i|O,\lambda)$（时刻t处于状态i的概率）
- $\xi_t(i,j)=P(q_t=i,q_{t+1}=j|O,\lambda)$（时刻t状态i转移到j的概率）

**M-Step**：重估参数
$$
\begin{aligned}
\bar{a}_{ij} &= \frac{\sum_t \xi_t(i,j)}{\sum_t \gamma_t(i)} \\
\bar{b}_j(k) &= \frac{\sum_{t:o_t=k} \gamma_t(j)}{\sum_t \gamma_t(j)} \\
\bar{\pi}_i &= \gamma_1(i)
\end{aligned}
$$

## 3. 前向-后向算法的角色

前向-后向算法是Baum-Welch的核心组件：

**前向概率**（$\alpha$）：
$$
\alpha_t(i) = P(o_1,...,o_t,q_t=i|\lambda)
$$
递归计算：
$$
\alpha_{t+1}(j) = \left[ \sum_{i=1}^N \alpha_t(i)a_{ij} \right] b_j(o_{t+1})
$$

**后向概率**（$\beta$）：
$$
\beta_t(i) = P(o_{t+1},...,o_T|q_t=i,\lambda)
$$
递归计算：
$$
\beta_t(i) = \sum_{j=1}^N a_{ij}b_j(o_{t+1})\beta_{t+1}(j)
$$

```python
def forward_algorithm(O, A, B, pi):
    T = len(O)
    N = A.shape[0]
    alpha = np.zeros((T, N))
    
    # 初始化
    alpha[0] = pi * B[:, O[0]]
    
    # 递归
    for t in range(1, T):
        for j in range(N):
            alpha[t,j] = np.sum(alpha[t-1] * A[:,j]) * B[j, O[t]]
    
    return alpha

def backward_algorithm(O, A, B):
    T = len(O)
    N = A.shape[0]
    beta = np.zeros((T, N))
    
    # 初始化
    beta[-1] = 1
    
    # 递归
    for t in range(T-2, -1, -1):
        for i in range(N):
            beta[t,i] = np.sum(A[i,:] * B[:, O[t+1]] * beta[t+1,:])
    
    return beta
```

## 4. Baum-Welch的迭代优化过程

完整算法流程：

1. 初始化参数$\lambda^{(0)}$
2. 迭代直到收敛：
   - E-Step：用当前$\lambda^{(k)}$计算$\gamma$和$\xi$
   - M-Step：用$\gamma,\xi$更新得到$\lambda^{(k+1)}$
3. 返回最终参数$\lambda^*$

收敛性证明依赖于EM算法的性质：每次迭代保证$P(O|\lambda^{(k+1)}) \geq P(O|\lambda^{(k)})$。

```python
def baum_welch(O, A, B, pi, max_iter=100):
    N = A.shape[0]
    T = len(O)
    
    for _ in range(max_iter):
        # E-Step
        alpha = forward_algorithm(O, A, B, pi)
        beta = backward_algorithm(O, A, B)
        gamma = alpha * beta / np.sum(alpha * beta, axis=1, keepdims=True)
        
        xi = np.zeros((T-1, N, N))
        for t in range(T-1):
            xi[t] = alpha[t,:,None] * A * B[:, O[t+1]][None,:] * beta[t+1][None,:]
            xi[t] /= np.sum(xi[t])
        
        # M-Step
        A = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0)[:, None]
        for k in range(B.shape[1]):
            B[:,k] = np.sum(gamma[O==k], axis=0) / np.sum(gamma, axis=0)
        pi = gamma[0]
    
    return A, B, pi
```

## 章节结论

Baum-Welch算法通过EM框架实现了HMM的无监督学习，其中前向-后向算法高效完成了概率推断。这种迭代优化方法使HMM能自动从观测数据中学习模式，是统计语言建模的核心技术之一。下一章将探讨解码问题——如何利用学习到的模型找到最优状态序列。
```