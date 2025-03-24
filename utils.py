# utils.py  
import time  
import numpy as np  
import matplotlib.pyplot as plt  
from sys import getsizeof  
import gc  
import torch  
import torch.nn as nn
from collections import defaultdict  
from typing import Dict, List  

from models import TestModel, TestModelForCausalLM

from load import get_qa_dataloader, QADataset, get_qa_dataset

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
    
    
    
    
    
class TorchOptimizerMonitor:  
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer):  
        self.model = model  
        self.optimizer = optimizer  
        self.history = []  
        self.start_time = time.time()  
        
        # 初始化CUDA内存基准  
        self._init_memory_stats()  
        
        # 注册梯度钩子  
        self.gradient_norms = defaultdict(list)  
        self._register_hooks()  

    def _init_memory_stats(self):  
        """初始化显存统计基准"""  
        torch.cuda.reset_peak_memory_stats()  
        self.base_mem = torch.cuda.memory_allocated()  
        self.optimizer_mem = 0  
        
    def _register_hooks(self):  
        """为模型参数注册梯度钩子"""  
        for name, param in self.model.named_parameters():  
            if param.requires_grad:  
                param.register_hook(  
                    lambda grad, name=name: self._gradient_hook(grad, name)  
                )  

    def _gradient_hook(self, grad: torch.Tensor, param_name: str):  
        """梯度钩子记录梯度统计"""  
        if grad is not None:  
            self.gradient_norms[param_name].append(grad.norm().item())  

    def _get_optimizer_states_mem(self) -> Dict[str, int]:  
        """计算优化器状态内存用量"""  
        state_mem = {}  
        for state in self.optimizer.state.values():  
            # 每个参数的状态字典可能包含多个变量（如Adam优化器中的一阶动量m和二阶动量v）
            for k, v in state.items():  
                if isinstance(v, torch.Tensor): 
                    # 累加相同状态名的内存用量 
                    state_mem[k] = state_mem.get(k, 0) + v.element_size() * v.nelement()  
        return state_mem  

    def record(self, loss: torch.Tensor = None):  
        """记录当前训练状态"""  
        torch.cuda.synchronize()  
        record = {  
            "timestamp": time.time() - self.start_time,  
            "step": len(self.history) + 1,  
            "loss": loss.item() if loss else None,  
            "learning_rate": self._get_current_lr(),  
            "memory": self._get_memory_stats(),  
            "grad_stats": self._get_grad_stats(),  
            "optimizer_states": self._get_optimizer_states_mem()  
        }  
        self.history.append(record)  
        return record  

    def _get_current_lr(self) -> float:  
        """获取当前学习率"""  
        return self.optimizer.param_groups[0]['lr']  

    def _get_memory_stats(self) -> Dict[str, int]:  
        """获取显存统计信息"""  
        return {  
            "allocated": torch.cuda.memory_allocated() - self.base_mem,  
            "peak": torch.cuda.max_memory_allocated() - self.base_mem,  
            "model_params": sum(p.element_size() * p.nelement()   
                              for p in self.model.parameters()),  
            "gradients": sum(p.grad.element_size() * p.grad.nelement()   
                           for p in self.model.parameters()   
                           if p.grad is not None),  
        }  

    def _get_grad_stats(self) -> Dict[str, float]:  
        """获取梯度统计信息"""  
        grads = []  
        for param in self.model.parameters():  
            if param.grad is not None:  
                grads.append(param.grad.detach().cpu().view(-1))  
        
        if not grads:  
            return {}  
            
        all_grads = torch.cat(grads)  
        return {  
            "norm": all_grads.norm().item(),  
            "mean": all_grads.mean().item(),  
            "std": all_grads.std().item(),  
            "nan_count": torch.isnan(all_grads).sum().item(),  
            "inf_count": torch.isinf(all_grads).sum().item(),  
        }  

    def plot_metrics(self, save_path: str = "training_metrics.png"):  
        """生成训练指标可视化图表"""  
        plt.figure(figsize=(18, 12))  
        
        # Loss曲线  
        plt.subplot(2, 3, 1)  
        losses = [r['loss'] for r in self.history if r['loss']]  
        plt.plot(losses, label='Training Loss')  
        plt.title("Loss Curve")  
        plt.xlabel("Step")  
        plt.legend()  

        # 学习率变化  
        plt.subplot(2, 3, 2)  
        lrs = [r['learning_rate'] for r in self.history]  
        plt.plot(lrs, color='orange')  
        plt.title("Learning Rate Schedule")  
        plt.xlabel("Step")  

        # 显存使用  
        plt.subplot(2, 3, 3)  
        mem_alloc = [r['memory']['allocated']/1e9 for r in self.history]  
        mem_peak = [r['memory']['peak']/1e9 for r in self.history]  
        plt.plot(mem_alloc, label='Current Memory')  
        plt.plot(mem_peak, label='Peak Memory')  
        plt.title("GPU Memory Usage (GB)")  
        plt.legend()  

        # 梯度统计  
        plt.subplot(2, 3, 4)  
        grad_norms = [r['grad_stats']['norm']/1e6 for r in self.history   
                     if 'norm' in r['grad_stats']]  
        plt.plot(grad_norms, color='green')  
        plt.title("Gradient Norm (M)")  

        # 优化器状态内存  
        plt.subplot(2, 3, 5)  
        state_names = list(self.history[0]['optimizer_states'].keys())  
        for state in state_names:  
            state_mem = [r['optimizer_states'][state]/1e6 for r in self.history]  
            plt.plot(state_mem, label=state)  
        plt.title("Optimizer States Memory (MB)")  
        plt.legend()  

        # 参数梯度分布  
        plt.subplot(2, 3, 6)  
        param_types = ['embeddings', 'attention', 'feed_forward']  
        for pt in param_types:  
            norms = [n[-1] if n else 0   
                    for n in self.gradient_norms.values() if pt in n]  
            plt.hist(norms, bins=50, alpha=0.5, label=pt)  
        plt.yscale('log')  
        plt.title("Gradient Distribution by Layer Type")  
        plt.legend()  

        plt.tight_layout()  
        plt.savefig(save_path)  
        plt.close()  

    def generate_report(self) -> str:  
        """生成训练分析报告"""  
        last = self.history[-1]  
        report = [  
            "===== Training Analysis Report =====",  
            f"Total Steps: {len(self.history)}",  
            f"Total Time: {last['timestamp']:.2f}s",  
            f"Final Loss: {last['loss']:.4f}",  
            f"Peak GPU Memory: {last['memory']['peak']/1e9:.2f} GB",  
            f"Final Gradient Norm: {last['grad_stats']['norm']/1e6:.2f} M",  
            "\n--- Memory Breakdown ---",  
            f"Model Parameters: {last['memory']['model_params']/1e9:.2f} GB",  
            f"Gradients: {last['memory']['gradients']/1e9:.2f} GB",  
            f"Optimizer States: {sum(last['optimizer_states'].values())/1e9:.2f} GB"  
        ]  
        return "\n".join(report) 
    
    


# 示例用法（与Hugging Face Trainer集成）  
def test_pytorch_optimizer():  
    from transformers import Trainer, TrainingArguments  
    
    class MonitoringTrainer(Trainer):  
        def __init__(self, *args, **kwargs):  
            super().__init__(*args, **kwargs)  
            self.monitor = None  
            
        def training_step(self, model, inputs):  
            loss = super().training_step(model, inputs)  
            if self.monitor is None:  
                self.monitor = TorchOptimizerMonitor(model, self.optimizer)  
            self.monitor.record(loss)  
            return loss  
    
    # 在训练参数中配置  
    training_args = TrainingArguments(  
        output_dir='./output',  
        per_device_train_batch_size=8,  
        logging_steps=50,  
    )  
    
    model = TestModelForCausalLM()
    
    # dataset = get_qa_dataloader()
    dataset = get_qa_dataset()
    # 使用自定义Trainer  
    trainer = MonitoringTrainer(  
        model=model,  
        args=training_args,  
        train_dataset=dataset,  
    )  
    
    # 开始训练  
    trainer.train()  
    
    # 生成报告和图表  
    trainer.monitor.plot_metrics()  
    print(trainer.monitor.generate_report())  
    
    
    




def test_my_optimizer():
    
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
    
    


if __name__ == "__main__":
    # test_my_optimizer()
    test_pytorch_optimizer()