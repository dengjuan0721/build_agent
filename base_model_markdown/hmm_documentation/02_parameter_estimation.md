```markdown
# Parameter Estimation

## 1. HMM参数估计方法概述

隐马尔可夫模型(HMM)的参数估计主要通过以下三类参数定义：
- 初始状态概率分布：π = (π₁, ..., π_N)，π_i = P(q₁ = s_i)
- 状态转移矩阵：A = [a_{ij}]，a_{ij} = P(q_{t+1}=s_j | q_t=s_i)
- 观测概率矩阵：B = [b_j(k)]，b_j(k) = P(o_t=v_k | q_t=s_j)

参数估计的核心目标是找到使观测序列似然最大化的参数组合：

$$
θ^* = \underset{θ}{\arg\max} P(O|θ)
$$

## 2. 最大似然估计在HMM中的应用

### 2.1 监督学习场景
当拥有标注数据（已知状态序列）时，可直接使用频率估计：

```python
# 状态转移概率估计示例
def estimate_transition(states):
    count = np.zeros((N,N))
    for t in range(len(states)-1):
        i, j = states[t], states[t+1]
        count[i,j] += 1
    return count / count.sum(axis=1, keepdims=True)
```

数学表达：
$$
\hat{a}_{ij} = \frac{N_{ij}}{\sum_{k=1}^N N_{ik}}
$$

### 2.2 非监督学习场景
当仅有观测序列时，需采用EM算法迭代优化：

```python
def forward_backward(O, π, A, B):
    # 前向算法计算α
    α = np.zeros((T,N))
    α[0] = π * B[:,O[0]]
    for t in range(1,T):
        α[t] = (α[t-1] @ A) * B[:,O[t]]
    
    # 后向算法计算β
    β = np.ones((T,N))
    for t in range(T-2,-1,-1):
        β[t] = A @ (B[:,O[t+1]] * β[t+1])
    
    return α, β
```

## 3. 监督学习与非监督学习的区别

| 特征        | 监督学习                  | 非监督学习                |
|-------------|--------------------------|--------------------------|
| 数据要求    | 需要标注的状态序列        | 仅需观测序列             |
| 典型方法    | 频率统计                  | Baum-Welch/EM算法        |
| 收敛速度    | 单次计算                  | 迭代收敛                 |
| 应用场景    | 语音识别（已知音素边界）  | 基因序列分析             |

## 章节结论

本节系统梳理了HMM参数估计的两大范式：基于标注数据的监督学习和仅依赖观测序列的非监督学习。最大似然估计框架为两种场景提供了统一的理论基础，其中EM算法通过迭代下界优化解决了隐变量导致的估计难题。这些方法为后续深入讨论Baum-Welch算法奠定了必要的理论基础。
```