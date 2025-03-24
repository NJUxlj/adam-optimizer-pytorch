import numpy as np  
import torch
import torch.nn as nn
import os

class AdafactorOptimizer:  
    def __init__(self, params, lr=1e-3, beta2=0.999,   
                 eps_scale=1e-3, decay_rate=-0.8,   
                 weight_decay=0.0, clip_threshold=1.0):  
        self.params = params  
        self.lr = lr  
        self.beta2 = beta2  
        self.eps_scale = eps_scale  
        self.decay_rate = decay_rate  
        self.weight_decay = weight_decay  
        self.clip_threshold = clip_threshold  
        self.r = np.zeros_like(params)  # 行方向RMS  
        self.c = np.zeros_like(params)  # 列方向RMS  
        self.v = np.zeros_like(params)  # 二阶动量近似  
    
    def _rms(self, x):  
        return np.sqrt(np.mean(x**2))  
    
    def step(self, grads):  
        # 梯度裁剪  
        grad_norm = self._rms(grads)  
        clip_factor = np.minimum(1.0, self.clip_threshold / grad_norm)  
        grads *= clip_factor  
        
        # 更新二阶动量估计  
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads**2  
        
        # 矩阵分解（简化版，实际应为行和列分开计算）  
        r_factor = np.mean(self.v, keepdims=True)  
        c_factor = np.mean(self.v, keepdims=True)  
        
        # 计算相对量级  
        eps = self.eps_scale * np.maximum(self.eps_scale, self._rms(self.params))  
        
        # 参数更新量计算  
        update = grads / (np.sqrt(r_factor * c_factor) + eps)  
        
        # 学习率缩放  
        param_scale = np.maximum(eps, self._rms(self.params))  
        lr_scaled = self.lr * param_scale  
        
        # 应用更新  
        self.params -= lr_scaled * update  
        
        # 权重衰减（解耦）  
        if self.weight_decay > 0:  
            self.params -= self.lr * self.weight_decay * self.params  
        
        return self.params  

# 测试用例（与Adam相同的优化目标）  
if __name__ == "__main__":  
    x = np.array([3.0], dtype=np.float32)  
    adafactor = AdafactorOptimizer(x, lr=0.5, weight_decay=0.01)  
    
    for epoch in range(100):  
        grad = 4 * x**3 + 9 * x**2  # 梯度计算  
        x = adafactor.step(grad)  
        
        if epoch % 10 == 0:  
            loss = x[0]**4 + 3*x[0]**3 + 2  
            print(f"Epoch {epoch}: x = {x[0]:.4f}, loss = {loss:.2f}")  
