
import time  
import numpy as np  
import gc  
import torch  
import torch.nn as nn
from collections import defaultdict  
from typing import Dict, List  

from models import TestModel, TestModelForCausalLM

from load import get_qa_dataloader, QADataset, get_qa_dataset

from utils import TorchOptimizerMonitor, OptimizerMonitor



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        dataloader_pin_memory=False,
    )  
    
    
    
    model = TestModelForCausalLM().to(DEVICE)
    
    
    
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