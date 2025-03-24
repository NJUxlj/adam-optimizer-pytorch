import torch
import os
import torch.nn as nn
import numpy as np
import random


class AdamOptimizer:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros_like(params)
        self.v = np.zeros_like(params)
        self.t = 0

    def step(self, grads):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        
        # 偏差校正
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # 参数更新
        self.params -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return self.params

# 测试用例（最小化 f(x) = x^4 + 3x^3 + 2）
if __name__ == "__main__":
    # 初始化参数和优化器
    x = np.array([3.0], dtype=np.float32) # Weights
    adam = AdamOptimizer(x, lr=0.1)
    
    # 训练循环
    for epoch in range(100):
        grad = 4 * x**3 + 9 * x**2  # 手动计算梯度
        x = adam.step(grad)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: x = {x[0]:.4f}, f(x) = {x[0]**4 + 3*x[0]**3 + 2:.2f}")
