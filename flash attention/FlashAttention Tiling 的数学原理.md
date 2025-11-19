我来详细解释 FlashAttention 使用 Tiling 的数学原理。这是一个非常精妙的设计！

## FlashAttention Tiling 的数学原理

### 一、标准 Attention 的数学形式

标准的 Attention 计算公式：

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

展开为三步：

1. **计算注意力分数**：$S = QK^T \in \mathbb{R}^{N \times N}$
2. **归一化**：$P = \text{Softmax}(S) \in \mathbb{R}^{N \times N}$
3. **加权求和**：$O = PV \in \mathbb{R}^{N \times d}$

**问题**：中间矩阵 $S$ 和 $P$ 的大小是 $O(N^2)$，无法放入 SRAM。

---

### 二、Softmax 的数学性质

FlashAttention 的关键在于利用 Softmax 的**可分解性**。

#### 2.1 标准 Softmax

对于向量 $x = [x_1, x_2, ..., x_N]$：

$$
\text{Softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^N e^{x_j}}
$$

#### 2.2 数值稳定的 Softmax

为了避免数值溢出，减去最大值：

$$
\text{Softmax}(x)_i = \frac{e^{x_i - m}}{\sum_{j=1}^N e^{x_j - m}}
$$

其中 $m = \max(x)$

#### 2.3 Softmax 的分块计算（关键！）

假设将向量 $x$ 分成两块：$x = [x^{(1)}, x^{(2)}]$

**问题**：如何从 $\text{Softmax}(x^{(1)})$ 和 $\text{Softmax}(x^{(2)})$ 计算出 $\text{Softmax}(x)$？

**答案**：使用在线 Softmax（Online Softmax）算法！

---

### 三、在线 Softmax（Online Softmax）

这是 FlashAttention 的核心数学技巧。

#### 3.1 问题设定

已知：
- 第一块的统计量：$m^{(1)} = \max(x^{(1)})$，$\ell^{(1)} = \sum_{j} e^{x_j^{(1)} - m^{(1)}}$
- 第二块的统计量：$m^{(2)} = \max(x^{(2)})$，$\ell^{(2)} = \sum_{j} e^{x_j^{(2)} - m^{(2)}}$

求：整体的 Softmax

#### 3.2 更新公式

**步骤 1：更新全局最大值**

$$
m^{\text{new}} = \max(m^{(1)}, m^{(2)})
$$

**步骤 2：更新归一化因子**

$$
\ell^{\text{new}} = e^{m^{(1)} - m^{\text{new}}} \cdot \ell^{(1)} + e^{m^{(2)} - m^{\text{new}}} \cdot \ell^{(2)}
$$

**步骤 3：更新输出**

如果已经计算了部分输出 $O^{(1)} = \text{Softmax}(x^{(1)}) \cdot V^{(1)}$，则：

$$
O^{\text{new}} = \frac{e^{m^{(1)} - m^{\text{new}}} \cdot \ell^{(1)}}{\ell^{\text{new}}} \cdot O^{(1)} + \frac{e^{m^{(2)} - m^{\text{new}}} \cdot \ell^{(2)}}{\ell^{\text{new}}} \cdot O^{(2)}
$$

#### 3.3 数学推导

让我们证明这个公式的正确性：

```
目标：计算 Softmax([x^(1), x^(2)])

标准 Softmax:
  Softmax(x_i) = exp(x_i - m) / ℓ
  其中 m = max(x), ℓ = Σ exp(x_j - m)

对于第一块:
  Softmax(x_i^(1)) = exp(x_i^(1) - m^(1)) / ℓ^(1)

对于完整向量:
  Softmax(x_i^(1)) = exp(x_i^(1) - m^new) / ℓ^new

关键：建立两者的关系
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

exp(x_i^(1) - m^new) / ℓ^new
= exp(x_i^(1) - m^(1)) · exp(m^(1) - m^new) / ℓ^new
= [exp(x_i^(1) - m^(1)) / ℓ^(1)] · [ℓ^(1) · exp(m^(1) - m^new) / ℓ^new]
= Softmax_old(x_i^(1)) · [ℓ^(1) · exp(m^(1) - m^new) / ℓ^new]

其中:
  ℓ^new = exp(m^(1) - m^new) · ℓ^(1) + exp(m^(2) - m^new) · ℓ^(2)

这样就可以增量更新 Softmax！
```

---

### 四、FlashAttention 的分块算法

#### 4.1 符号定义

- $Q, K, V \in \mathbb{R}^{N \times d}$：输入矩阵
- $B_r, B_c$：块大小（行块和列块）
- $T_r = \lceil N / B_r \rceil$：行块数量
- $T_c = \lceil N / B_c \rceil$：列块数量

将矩阵分块：
- $Q = [Q_1, Q_2, ..., Q_{T_r}]$，每块 $Q_i \in \mathbb{R}^{B_r \times d}$
- $K = [K_1, K_2, ..., K_{T_c}]$，每块 $K_j \in \mathbb{R}^{B_c \times d}$
- $V = [V_1, V_2, ..., V_{T_c}]$，每块 $V_j \in \mathbb{R}^{B_c \times d}$

#### 4.2 算法流程

```
FlashAttention 前向传播算法
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

输入: Q, K, V ∈ ℝ^(N×d) 存储在 HBM
输出: O ∈ ℝ^(N×d) 存储在 HBM

1. 将 Q 分成 Tr 块: Q₁, Q₂, ..., Q_Tr (每块 Br×d)
2. 将 K, V 分成 Tc 块: K₁, K₂, ..., K_Tc 和 V₁, V₂, ..., V_Tc (每块 Bc×d)

3. 初始化输出 O = 0 ∈ ℝ^(N×d) 在 HBM 中

4. for i = 1 to Tr do:  // 遍历 Q 的每一块
   
   a) 从 HBM 加载 Qᵢ 到 SRAM
   
   b) 初始化:
      - Oᵢ = 0 ∈ ℝ^(Br×d)  (在 SRAM 中)
      - ℓᵢ = 0 ∈ ℝ^Br       (归一化因子)
      - mᵢ = -∞ ∈ ℝ^Br      (最大值)
   
   c) for j = 1 to Tc do:  // 遍历 K, V 的每一块
      
      i.  从 HBM 加载 Kⱼ, Vⱼ 到 SRAM
      
      ii. 计算注意力分数:
          Sᵢⱼ = Qᵢ Kⱼᵀ ∈ ℝ^(Br×Bc)  (在 SRAM 中)
      
      iii. 计算当前块的统计量:
           m̃ᵢⱼ = rowmax(Sᵢⱼ) ∈ ℝ^Br  (每行的最大值)
           P̃ᵢⱼ = exp(Sᵢⱼ - m̃ᵢⱼ) ∈ ℝ^(Br×Bc)  (未归一化的 Softmax)
           ℓ̃ᵢⱼ = rowsum(P̃ᵢⱼ) ∈ ℝ^Br  (每行的和)
      
      iv. 更新全局统计量:
          m_new = max(mᵢ, m̃ᵢⱼ)
          ℓ_new = exp(mᵢ - m_new)·ℓᵢ + exp(m̃ᵢⱼ - m_new)·ℓ̃ᵢⱼ
      
      v.  更新输出:
          Oᵢ = exp(mᵢ - m_new)·Oᵢ + exp(m̃ᵢⱼ - m_new)·P̃ᵢⱼVⱼ
          Oᵢ = Oᵢ / ℓ_new  (归一化)
      
      vi. 更新统计量:
          mᵢ = m_new
          ℓᵢ = ℓ_new
   
   d) 将 Oᵢ 从 SRAM 写回 HBM
```

#### 4.3 数学公式详解

**关键公式 1：注意力分数块**

$$
S_{ij} = Q_i K_j^T \in \mathbb{R}^{B_r \times B_c}
$$

这是一个**小矩阵**，可以放入 SRAM！

**关键公式 2：块内 Softmax**

$$
\tilde{m}_{ij} = \max_{k} (S_{ij})_{:,k} \in \mathbb{R}^{B_r}
$$

$$
\tilde{P}_{ij} = \exp(S_{ij} - \tilde{m}_{ij}) \in \mathbb{R}^{B_r \times B_c}
$$

$$
\tilde{\ell}_{ij} = \sum_{k} (\tilde{P}_{ij})_{:,k} \in \mathbb{R}^{B_r}
$$

**关键公式 3：统计量更新**

$$
m_i^{\text{new}} = \max(m_i^{\text{old}}, \tilde{m}_{ij})
$$

$$
\ell_i^{\text{new}} = e^{m_i^{\text{old}} - m_i^{\text{new}}} \cdot \ell_i^{\text{old}} + e^{\tilde{m}_{ij} - m_i^{\text{new}}} \cdot \tilde{\ell}_{ij}
$$

**关键公式 4：输出更新**

$$
O_i^{\text{new}} = \text{diag}\left(\frac{e^{m_i^{\text{old}} - m_i^{\text{new}}} \cdot \ell_i^{\text{old}}}{\ell_i^{\text{new}}}\right) O_i^{\text{old}} + \text{diag}\left(\frac{e^{\tilde{m}_{ij} - m_i^{\text{new}}} \cdot \tilde{\ell}_{ij}}{\ell_i^{\text{new}}}\right) \tilde{P}_{ij} V_j
$$

简化形式：

$$
O_i = \frac{e^{m_i^{\text{old}} - m_i^{\text{new}}} \cdot \ell_i^{\text{old}}}{\ell_i^{\text{new}}} \cdot O_i + \frac{e^{\tilde{m}_{ij} - m_i^{\text{new}}}}{\ell_i^{\text{new}}} \cdot \tilde{P}_{ij} V_j
$$

---

### 五、具体计算示例

让我们用一个小例子来演示：

#### 5.1 问题设定

```
N = 4, d = 2
块大小: Br = Bc = 2

Q = [q₁]   K = [k₁]   V = [v₁]
    [q₂]       [k₂]       [v₂]
    [q₃]       [k₃]       [v₃]
    [q₄]       [k₄]       [v₄]

分块:
Q₁ = [q₁, q₂]ᵀ,  Q₂ = [q₃, q₄]ᵀ
K₁ = [k₁, k₂]ᵀ,  K₂ = [k₃, k₄]ᵀ
V₁ = [v₁, v₂]ᵀ,  V₂ = [v₃, v₄]ᵀ
```

#### 5.2 计算 O₁（Q₁ 对应的输出）

**迭代 1：处理 K₁, V₁**

```
1. 加载 Q₁, K₁, V₁ 到 SRAM

2. 计算 S₁₁ = Q₁K₁ᵀ:
   S₁₁ = [q₁·k₁  q₁·k₂]  ∈ ℝ^(2×2)
         [q₂·k₁  q₂·k₂]

3. 计算统计量:
   m̃₁₁ = [max(q₁·k₁, q₁·k₂)]  ∈ ℝ²
          [max(q₂·k₁, q₂·k₂)]
   
   P̃₁₁ = exp(S₁₁ - m̃₁₁)
   
   ℓ̃₁₁ = rowsum(P̃₁₁)

4. 初始化 (第一次迭代):
   m₁ = m̃₁₁
   ℓ₁ = ℓ̃₁₁
   O₁ = P̃₁₁ V₁ / ℓ̃₁₁
```

**迭代 2：处理 K₂, V₂**

```
1. 加载 Q₁, K₂, V₂ 到 SRAM (Q₁ 可能还在)

2. 计算 S₁₂ = Q₁K₂ᵀ:
   S₁₂ = [q₁·k₃  q₁·k₄]  ∈ ℝ^(2×2)
         [q₂·k₃  q₂·k₄]

3. 计算当前块统计量:
   m̃₁₂ = rowmax(S₁₂)
   P̃₁₂ = exp(S₁₂ - m̃₁₂)
   ℓ̃₁₂ = rowsum(P̃₁₂)

4. 更新全局统计量:
   m_new = max(m₁, m̃₁₂)  (逐元素)
   
   ℓ_new = exp(m₁ - m_new)·ℓ₁ + exp(m̃₁₂ - m_new)·ℓ̃₁₂

5. 更新输出:
   O₁_new = [exp(m₁ - m_new)·ℓ₁ / ℓ_new] · O₁
          + [exp(m̃₁₂ - m_new) / ℓ_new] · P̃₁₂V₂

6. 更新状态:
   m₁ = m_new
   ℓ₁ = ℓ_new
   O₁ = O₁_new

7. 写回 O₁ 到 HBM
```

#### 5.3 数值示例

```
假设具体数值:

Q₁ = [1, 0]ᵀ,  K₁ = [1, 0]ᵀ,  V₁ = [1, 0]ᵀ
     [0, 1]         [0, 1]         [0, 1]

K₂ = [0.5, 0.5]ᵀ,  V₂ = [0.5, 0.5]ᵀ
     [0.5, 0.5]         [0.5, 0.5]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

迭代 1:

S₁₁ = Q₁K₁ᵀ = [1  0]
              [0  1]

m̃₁₁ = [1]
       [1]

P̃₁₁ = exp(S₁₁ - m̃₁₁) = [1  0]
                         [0  1]

ℓ̃₁₁ = [1]
       [1]

O₁ = P̃₁₁V₁ / ℓ̃₁₁ = [1, 0]ᵀ
                     [0, 1]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

迭代 2:

S₁₂ = Q₁K₂ᵀ = [0.5  0.5]
              [0.5  0.5]

m̃₁₂ = [0.5]
       [0.5]

P̃₁₂ = exp(S₁₂ - m̃₁₂) = [1  1]
                         [1  1]

ℓ̃₁₂ = [2]
       [2]

更新:
m_new = max([1, 1], [0.5, 0.5]) = [1, 1]

ℓ_new = exp([1,1] - [1,1])·[1,1] + exp([0.5,0.5] - [1,1])·[2,2]
      = 1·[1,1] + exp(-0.5)·[2,2]
      = [1,1] + [1.21, 1.21]
      = [2.21, 2.21]

O₁_new = [1/2.21, 1/2.21] · [[1,0], [0,1]]
       + [exp(-0.5)/2.21, exp(-0.5)/2.21] · [[0.5,0.5], [0.5,0.5]]
       
       ≈ [0.45, 0] + [0.27, 0.27]
         [0, 0.45] + [0.27, 0.27]
       
       = [0.72, 0.27]
         [0.27, 0.72]

最终 O₁ = [0.72, 0.27]ᵀ
          [0.27, 0.72]
```

---

### 六、复杂度分析

#### 6.1 内存复杂度

**原始 Attention**：
- 需要存储 $S, P \in \mathbb{R}^{N \times N}$
- 内存：$O(N^2)$

**FlashAttention**：
- 只需要存储块：$S_{ij}, P_{ij} \in \mathbb{R}^{B_r \times B_c}$
- 统计量：$m_i, \ell_i \in \mathbb{R}^{B_r}$
- 内存：$O(B_r \times B_c) = O(N)$（当块大小固定时）

#### 6.2 IO 复杂度

**原始 Attention**：
```
HBM 访问次数:
1. 读 Q, K: 2Nd
2. 写 S: N²
3. 读 S: N²
4. 写 P: N²
5. 读 P, V: N² + Nd
6. 写 O: Nd

总计: O(N² + Nd) ≈ O(N²)
```

**FlashAttention**：
```
HBM 访问次数:
外层循环 (Tr 次):
  内层循环 (Tc 次):
    - 读 Qi: Br·d
    - 读 Kj, Vj: 2·Bc·d
    - 写 Oi: Br·d (最后)

总计: Tr·Tc·(Br·d + 2·Bc·d) + Tr·Br·d
    = (N/Br)·(N/Bc)·(Br·d + 2·Bc·d) + N·d
    ≈ O(N²d / (Br·Bc))

当 Br, Bc ~ √(SRAM_size / d) 时:
    IO 复杂度 = O(N²d² / SRAM_size)
```

**加速比**：

$$
\text{Speedup} = \frac{O(N^2)}{O(N^2 d^2 / M)} = \frac{M}{d^2}
$$

其中 $M$ 是 SRAM 大小。

对于 $M = 100$ KB, $d = 128$:
$$
\text{Speedup} = \frac{100 \times 1024}{128^2} \approx 6.25
$$

实际上由于其他优化（如 Kernel Fusion），加速比可达 2-4 倍。

---

### 七、为什么 Tiling 有效？

#### 7.1 数学上的可行性

**关键洞察**：Softmax 可以增量计算！

```
传统思维:
  必须先计算完整的 S = QKᵀ
  然后计算 Softmax(S)
  最后计算 O = Softmax(S)·V
  
FlashAttention 的突破:
  可以分块计算 Sij = QiKjᵀ
  可以增量更新 Softmax 的统计量
  可以增量更新输出 O
  
  不需要存储完整的 S 和 P！
```

#### 7.2 硬件上的优势

```
内存层次:
  SRAM: 19 TB/s, 20 MB
  HBM:  1.5 TB/s, 40 GB

策略:
  ✓ 小块 (Br×Bc) 放入 SRAM
  ✓ 在 SRAM 中完成所有计算
  ✓ 只在必要时访问 HBM
  
结果:
  ✓ 大部分时间在快速的 SRAM 中
  ✓ HBM 访问减少到 O(N²/M)
  ✓ 整体速度提升 2-4 倍
```

---

### 八、总结

#### FlashAttention Tiling 的三大数学支柱

1. **Softmax 的可分解性**
   - 可以从部分统计量重构完整 Softmax
   - 关键公式：$\ell^{\text{new}} = e^{m^{(1)} - m^{\text{new}}} \cdot \ell^{(1)} + e^{m^{(2)} - m^{\text{new}}} \cdot \ell^{(2)}$

2. **在线算法（Online Algorithm）**
   - 增量更新统计量 $(m, \ell)$
   - 增量更新输出 $O$
   - 不需要存储完整的中间结果

3. **分块计算（Tiling）**
   - 将 $O(N^2)$ 的矩阵分成 $O(B^2)$ 的小块
   - 小块可以放入 SRAM
   - 减少 HBM 访问到 $O(N^2/M)$

#### 核心公式总结

$$
\boxed{
\begin{aligned}
&\text{块内计算:} \\
&S_{ij} = Q_i K_j^T \\
&\tilde{m}_{ij} = \text{rowmax}(S_{ij}) \\
&\tilde{P}_{ij} = \exp(S_{ij} - \tilde{m}_{ij}) \\
&\tilde{\ell}_{ij} = \text{rowsum}(\tilde{P}_{ij}) \\
\\
&\text{统计量更新:} \\
&m^{\text{new}} = \max(m^{\text{old}}, \tilde{m}_{ij}) \\
&\ell^{\text{new}} = e^{m^{\text{old}} - m^{\text{new}}} \ell^{\text{old}} + e^{\tilde{m}_{ij} - m^{\text{new}}} \tilde{\ell}_{ij} \\
\\
&\text{输出更新:} \\
&O^{\text{new}} = \frac{e^{m^{\text{old}} - m^{\text{new}}} \ell^{\text{old}}}{\ell^{\text{new}}} O^{\text{old}} + \frac{e^{\tilde{m}_{ij} - m^{\text{new}}}}{\ell^{\text{new}}} \tilde{P}_{ij} V_j
\end{aligned}
}
$$

这就是 FlashAttention 的数学魔法！🎯