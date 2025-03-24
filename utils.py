# utils.py  
import time  
import numpy as np  
import matplotlib.pyplot as plt  
from sys import getsizeof  
import gc  
import torch  

class OptimizerMonitor:  
    def __init__(self, model=None, optimizer=None):  
        self.model = model  
        self.optimizer = optimizer  
        self.history = []  
        self.start_time = time.time()  
        self.mem_peak = 0  
        
        # CUDA内存基准（如果可用）  
        self.cuda_mem_base = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0  
        
    def _get_mem_usage(self, component):  
        """获取组件内存用量（支持PyTorch Tensor和Numpy数组）"""  
        if torch.is_tensor(component):  
            return component.element_size() * component.nelement()  
        elif isinstance(component, np.ndarray):  
            return component.nbytes  
        return getsizeof(component)  
    
    def record(self, grads, step_name="step"):  
        """记录单步监控数据"""  
        record = {  
            "timestamp": time.time() - self.start_time,  
            "step": len(self.history) + 1,  
            "step_name": step_name,  
            "grad_norm": np.linalg.norm(grads),  
            "grad_mean": np.mean(grads),  
            "grad_std": np.std(grads),  
            "mem_usage": {}  
        }  
        
        # 记录优化器状态内存  
        if self.optimizer:  
            opt_states = {}  
            for attr in dir(self.optimizer):  
                if attr.startswith('__') or callable(getattr(self.optimizer, attr)):  
                    continue  
                state = getattr(self.optimizer, attr)  
                if isinstance(state, (np.ndarray, torch.Tensor)):  
                    opt_states[attr] = self._get_mem_usage(state)  
            record["mem_usage"]["optimizer_states"] = opt_states  
        
        # 记录模型内存（需要PyTorch模型）  
        if self.model is not None:  
            model_mem = 0  
            param_mem = 0  
            grad_mem = 0  
            for param in self.model.parameters():  
                model_mem += self._get_mem_usage(param)  
                param_mem += self._get_mem_usage(param.data)  
                if param.grad is not None:  
                    grad_mem += self._get_mem_usage(param.grad)  
            record["mem_usage"].update({  
                "model_total": model_mem,  
                "parameters": param_mem,  
                "gradients": grad_mem  
            })  
        
        # 记录CUDA显存（如果可用）  
        if torch.cuda.is_available():  
            torch.cuda.synchronize()  
            current_mem = torch.cuda.memory_allocated() - self.cuda_mem_base  
            record["mem_usage"]["cuda_total"] = current_mem  
            self.mem_peak = max(self.mem_peak, current_mem)  
            record["mem_usage"]["cuda_peak"] = self.mem_peak  
        
        self.history.append(record)  
        return record  
    
    def plot_trends(self, save_path="optimizer_trends.png"):  
        """绘制训练趋势图"""  
        plt.figure(figsize=(15, 10))  
        
        # 梯度统计  
        plt.subplot(2, 2, 1)  
        plt.plot([h['grad_norm'] for h in self.history], label='Grad Norm')  
        plt.plot([h['grad_mean'] for h in self.history], label='Grad Mean')  
        plt.plot([h['grad_std'] for h in self.history], label='Grad Std')  
        plt.title("Gradient Statistics")  
        plt.legend()  
        
        # 内存使用  
        plt.subplot(2, 2, 2)  
        if self.model:  
            plt.plot([h['mem_usage']['parameters']/1e6 for h in self.history],   
                    label='Parameters (MB)')  
            plt.plot([h['mem_usage']['gradients']/1e6 for h in self.history],   
                    label='Gradients (MB)')  
        if 'cuda_total' in self.history[0]['mem_usage']:  
            plt.plot([h['mem_usage']['cuda_total']/1e6 for h in self.history],  
                    label='GPU Memory (MB)')  
        plt.title("Memory Usage")  
        plt.legend()  
        
        # 优化器状态（示例显示前3个状态）  
        plt.subplot(2, 2, 3)  
        if self.optimizer and 'optimizer_states' in self.history[0]['mem_usage']:  
            states = list(self.history[0]['mem_usage']['optimizer_states'].keys())[:3]  
            for state in states:  
                plt.plot([h['mem_usage']['optimizer_states'][state]/1e6 for h in self.history],  
                        label=f'{state} (MB)')  
            plt.title("Optimizer States Memory")  
            plt.legend()  
        
        # 时间线  
        plt.subplot(2, 2, 4)  
        plt.plot([h['timestamp'] for h in self.history],   
                [h['grad_norm'] for h in self.history], 'r-')  
        plt.xlabel('Time (s)')  
        plt.ylabel('Grad Norm')  
        plt.title("Gradient Norm Timeline")  
        
        plt.tight_layout()  
        plt.savefig(save_path)  
        plt.close()  
    
    def generate_report(self):  
        """生成文本报告"""  
        report = [  
            "===== Optimization Report =====",  
            f"Total Steps: {len(self.history)}",  
            f"Total Time: {self.history[-1]['timestamp']:.2f}s",  
            f"Peak GPU Memory: {self.history[-1]['mem_usage'].get('cuda_peak',0)/1e6:.2f} MB"   
        ]  
        
        if self.optimizer:  
            opt_states = self.history[0]['mem_usage']['optimizer_states']  
            report.append("\nOptimizer States Memory:")  
            for k, v in opt_states.items():  
                report.append(f"- {k}: {v/1e6:.2f} MB")  
        
        if self.model:  
            report.append("\nModel Memory Breakdown:")  
            report.append(f"- Parameters: {self.history[-1]['mem_usage']['parameters']/1e6:.2f} MB")  
            report.append(f"- Gradients: {self.history[-1]['mem_usage']['gradients']/1e6:.2f} MB")  
            report.append(f"- Total Model: {self.history[-1]['mem_usage']['model_total']/1e6:.2f} MB")  
        
        return "\n".join(report)  

 
if __name__ == "__main__":  
    # 示例模型和优化器  
    model = torch.nn.Linear(10, 2).cuda()  
    opt = torch.optim.Adam(model.parameters())  
    
    monitor = OptimizerMonitor(model=model, optimizer=opt)  
    
    # 模拟训练循环  
    for _ in range(100):  
        inputs = torch.randn(32, 10).cuda()  
        outputs = model(inputs)  
        loss = outputs.mean()  
        loss.backward()  
        
        # 记录前先获取梯度  
        grads = np.concatenate([p.grad.cpu().numpy().flatten() for p in model.parameters()])  
        monitor.record(grads)  
        
        opt.step()  
        opt.zero_grad()  
    
    # 生成报告和图表  
    monitor.plot_trends()  
    time.sleep(1)
    print(monitor.generate_report())  
    
    
    # 保存报告
    # with open("training_report.txt", "w") as f:  
    #     f.write(monitor.generate_report())  