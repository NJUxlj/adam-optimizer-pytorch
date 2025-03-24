# Adam-optimizer-pytorch
reproduce Adam with pytorch



## 大模型训练中常用的优化器


1. **AdamW[^1](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules)**
原理：
- 将权重衰减与梯度更新解耦
- 更新公式：

\[
\begin{aligned}
m_t = β1*m_{t-1} + (1-β1)*g_t
v_t = β2*v_{t-1} + (1-β2)*g_t²
θ_t = θ_{t-1} - η*(m_t/(√v_t + ε) + λθ_{t-1})
\end{aligned}
\]
  
- 优势：解决了Adam中L2正则化与权重衰减的等价性假设失效问题，更适合LLM微调

2. **Adafactor[^2](https://huggingface.co/docs/transformers/v4.40.0/en/main_classes/optimizer_schedules#transformers.Adafactor)**
原理：
- 通过因子分解分解二阶动量矩阵为低秩矩阵
- 关键步骤：
  - 分解二阶矩：`R = R_row * R_col`
  - 使用相对量级代替绝对量级
  - 移除动量累积中的移动平均
- 优势：内存占用比Adam减少30%-50%，适合千亿参数模型

3. **8-bit Adam[^3](https://huggingface.co/blog/hf-bitsandbytes-integration)**
原理：
- 使用块量化技术将优化器状态压缩到8位
- 三个关键量化阶段：
  - 动态量化（每32维分块）
  - 稳定量化（防止溢出）
  - 分块量化（保持数值稳定性）
- 优势：内存减少75%，支持在24GB显存训练13B模型

4. **Sophia[^4](https://arxiv.org/abs/2305.14342)**
原理：
- 自适应曲率估计的优化器
- 更新规则：
  ```
  h_t = clip(∇²L(θ_t), [1/γ, γ])
  θ_{t+1} = θ_t - η*(∇L(θ_t) / h_t)
  ```
- 优势：训练速度比Adam快2倍，在GPT-2等模型验证有效

5. **LAMB[^5](https://arxiv.org/abs/1904.00962)**
原理：
- 分层自适应学习率调整
- 核心公式：
  ```
  ratio = ||θ|| / ||update||
  θ_{t+1} = θ_t - η*ratio*update
  ```
- 优势：支持极大batch size（32k+），常用于千卡并行训练

6. **Shampoo[^6](https://arxiv.org/abs/2002.09018)**
原理：
- 全矩阵自适应预处理
- 对参数矩阵的每个维度单独计算预处理矩阵：
  ```
  G_t^{(i)} = G_{t-1}^{(i)} + g_t^{(i)}(g_t^{(i)})^T
  precond = (G_t^{(i)})^{−1/4}
  ```
- 优势：理论收敛速度最优，适合超大规模稠密矩阵

**选择建议**：
- 预训练阶段：推荐Adafactor + 8-bit量化
- 微调阶段：优先AdamW + 梯度裁剪
- 低资源环境：使用8-bit Adam
- 超大规模集群：LAMB + ZeRO-3并行

这些优化器在Hugging Face生态中均有现成实现，可通过`transformers.Trainer`直接调用。最新的优化趋势是结合二阶优化方法和量化技术，如Sophia优化器在LLaMA-2训练中已展现出显著优势。
