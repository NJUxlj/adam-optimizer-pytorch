import os
import torch
import torch.nn as nn
import numpy as np
import random



class AdamWOptimizer:  
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999,   
                 weight_decay=0.01, epsilon=1e-8):  
        self.params = params  
        self.lr = lr  
        self.beta1 = beta1  
        self.beta2 = beta2  
        self.weight_decay = weight_decay  
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
        
        # AdamW核心变化：解耦权重衰减  
        update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)  
        self.params -= update + self.lr * self.weight_decay * self.params  
        
        return self.params  

# 测试用例（带L2正则的优化目标 f(x) = x^4 + 3x^3 + 2 + 0.5x^2）  
if __name__ == "__main__":  
    # 初始化参数和优化器  
    x = np.array([3.0], dtype=np.float32)  
    adamw = AdamWOptimizer(x, lr=0.1, weight_decay=0.1)  
    
    # 训练循环  
    for epoch in range(100):  
        # 梯度计算（原始梯度 + L2正则梯度）  
        raw_grad = 4 * x**3 + 9 * x**2  
        l2_grad = 1.0 * x  # 对应0.5x^2项的梯度  
        total_grad = raw_grad + l2_grad  
        
        x = adamw.step(total_grad)  
        
        if epoch % 10 == 0:  
            loss = x[0]**4 + 3*x[0]**3 + 2 + 0.5*x[0]**2  
            print(f"Epoch {epoch}: x = {x[0]:.4f}, loss = {loss:.2f}")  
