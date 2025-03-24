# load.py  
import torch  
from torch.utils.data import Dataset, DataLoader  
from transformers import AutoTokenizer  
from collections import Counter  


MODEL_PATH = "D:\\models\\bert-base-chinese"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_qa_samples(num_samples=200):  
    """生成模拟QA数据集"""  
    samples = []  
    for i in range(num_samples):  
        # 生成不同主题的问题  
        subject = ["数学", "历史", "科学", "文学"][i % 4]  
        context = f"{subject}知识问答示例："  
        
        # 生成问题和答案对  
        if subject == "数学":  
            question = f"问题：{i+1}的平方根是多少？"  
            answer = f"答案：√{i+1} ≈ {round((i+1)**0.5, 2)}"  
        elif subject == "历史":  
            question = f"问题：公元{100+i}年发生的重要事件是？"  
            answer = "答案：示例历史事件（数据待补充）"  
        elif subject == "科学":  
            question = f"问题：元素周期表中第{i%10+1}号元素的名称是？"  
            answer = f"答案：{['氢','氦','锂','铍','硼','碳','氮','氧','氟','氖'][i%10]}"  
        else:  
            question = "问题：莎士比亚最著名的悲剧作品是？"  
            answer = "答案：《哈姆雷特》"  
            
        samples.append({  
            "id": str(i+1),  
            "context": context,  
            "question": question,  
            "answer": answer  
        })  
    return samples  

class QADataset(Dataset):  
    def __init__(self, samples, tokenizer, max_length=128):  
        self.tokenizer = tokenizer  
        self.max_length = max_length  
        self.inputs = []  
        self.targets = []  
        
        # 预处理所有样本  
        for sample in samples:  
            input_text = f"{sample['context']} {sample['question']}".strip()  
            target_text = sample['answer']  
            
            # Tokenize输入  
            tokenized_input = tokenizer(  
                input_text,  
                max_length=max_length,  
                truncation=True,  
                padding="max_length",  
                return_tensors="pt"  
            ).to(DEVICE)
            
            # Tokenize输出  
            tokenized_target = tokenizer(  
                target_text,  
                max_length=max_length,  # 答案通常较短  
                truncation=True,  
                padding="max_length",  
                return_tensors="pt"  
            ).to(DEVICE)  
            
            self.inputs.append(tokenized_input)  
            self.targets.append(tokenized_target)  
    
    def __len__(self):  
        return len(self.inputs)  
    
    def __getitem__(self, idx):  
        return {  
            "input_ids": torch.tensor(self.inputs[idx]["input_ids"]).to(DEVICE),  
            "attention_mask": torch.tensor(self.inputs[idx]["attention_mask"]).to(DEVICE),  
            "labels": torch.tensor(self.targets[idx]["input_ids"]).to(DEVICE) 
        }  

def collate_fn(batch):  
    """动态填充batch数据"""  
    def pad_sequence(sequences, padding_value):  
        return torch.nn.utils.rnn.pad_sequence(  
            sequences,  
            batch_first=True,  
            padding_value=padding_value  
        )  
    
    return {  
        "input_ids": pad_sequence([item["input_ids"] for item in batch], 0),  
        "attention_mask": pad_sequence([item["attention_mask"] for item in batch], 0),  
        "labels": pad_sequence([item["labels"] for item in batch], -100)  # 使用-100忽略loss计算  
    }  
    
    
def get_qa_dataset(
    model_name=MODEL_PATH,  
    batch_size=8,  
    max_length=128 
):
    # 初始化tokenizer  
    tokenizer = AutoTokenizer.from_pretrained(model_name)  
    
    # 生成样本数据  
    samples = generate_qa_samples()  
    
    # 创建数据集  
    dataset = QADataset(samples, tokenizer, max_length)  
    
    return dataset
    

def get_qa_dataloader(  
    model_name=MODEL_PATH,  
    batch_size=8,  
    max_length=128  
):  
    # 初始化tokenizer  
    tokenizer = AutoTokenizer.from_pretrained(model_name)  
    
    # 生成样本数据  
    samples = generate_qa_samples()  
    
    # 创建数据集  
    dataset = QADataset(samples, tokenizer, max_length)  
    
    # 创建DataLoader  
    return DataLoader(  
        dataset,  
        batch_size=batch_size,  
        collate_fn=collate_fn,  
        shuffle=True,  
        num_workers=2  
    )  

# 使用示例  
if __name__ == "__main__":  
    from transformers import AutoModelForSeq2SeqLM  
    
    # 测试数据加载  
    dataloader = get_qa_dataloader()  
    print(f"Total batches: {len(dataloader)}")  
    
    # 检查第一个batch  
    batch = next(iter(dataloader))  
    print("\nBatch结构：")  
    for k, v in batch.items():  
        print(f"{k}: {v.shape}")  
    
    # # 集成到Trainer的完整示例  
    # model = AutoModelForSeq2SeqLM.from_pretrained("bert-base-chinese")  
    
    # training_args = TrainingArguments(  
    #     output_dir="./results",  
    #     per_device_train_batch_size=8,  
    #     num_train_epochs=3,  
    #     logging_steps=10,  
    # )  
    
    # train_dataset = QADataset(generate_qa_samples(),   
    #                          AutoTokenizer.from_pretrained("bert-base-chinese"))  
    
    # trainer = MonitoringTrainer(  
    #     model=model,  
    #     args=training_args,  
    #     train_dataset=train_dataset,  # 实际使用需转换为datasets.Dataset格式  
    #     data_collator=collate_fn  
    # )  