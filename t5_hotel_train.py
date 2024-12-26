from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader, Dataset
import torch, os
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup  # 导入学习率调度器

# 加载 T5 模型
model_name = "./models/mengzi-t5-base"
hotel_model_name = "models/t5_hotel_model.pth"
hotel_optimizer_name = "models/t5_hotel_optimizer.pth"

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 数据集
import json

with open('QA.base.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 数据预处理
inputs = [f"Question: {item['question']} Answer:" for item in data]
labels = [item['answer'] for item in data]

encoded_inputs = tokenizer(inputs, max_length=256, padding=True, truncation=True, return_tensors="pt")
encoded_labels = tokenizer(labels, max_length=256, padding=True, truncation=True, return_tensors="pt")
encoded_labels['input_ids'][encoded_labels['input_ids'] == tokenizer.pad_token_id] = -100

# 数据加载
class QADataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs['input_ids'])

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs['input_ids'][idx],
            "attention_mask": self.inputs['attention_mask'][idx],
            "labels": self.labels['input_ids'][idx],
        }

dataset = QADataset(encoded_inputs, encoded_labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)  # 设置 num_workers 为所需的线程数

# 优化器
optimizer = AdamW(model.parameters(), lr=5e-5)
# 学习率调度器
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * 35)

# 训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Starting train by {device}")
model.to(device)
model.train()

# 判断模型参数文件是否存在，如果存在则加载
if os.path.exists(hotel_model_name):
    model.load_state_dict(torch.load(hotel_model_name))  # 加载保存的模型参数
# 判断优化器状态文件是否存在，如果存在则加载
if os.path.exists(hotel_optimizer_name):
    optimizer.load_state_dict(torch.load(hotel_optimizer_name))  # 加载优化器状态

# 继续训练
best_loss = float('inf')  # 初始化最佳损失
patience = 3  # 早停法的耐心轮数
patience_counter = 0  # 计数器

for epoch in range(30):  # 继续训练的轮数
    total_loss = 0
    model.train()  # 确保模型在训练模式
    for batch in dataloader:
        batch = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()  # 更新学习率
        optimizer.zero_grad()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    # 早停法
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        # 保存最佳模型
        torch.save(model.state_dict(), hotel_model_name)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break  # 触发早停法，停止训练

# 训练结束后保存模型和优化器状态
torch.save(model.state_dict(), hotel_model_name)  # 保存模型参数
torch.save(optimizer.state_dict(), hotel_optimizer_name)  # 保存优化器状态

# 推理
model.eval()
# 加载模型参数
model.load_state_dict(torch.load(hotel_model_name))  # 加载保存的模型参数
test_question = "酒店有泳池吗？"
input_text = f"Question: {test_question} Answer:"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

# 模型生成答案
output_ids = model.generate(input_ids, max_length=150)
answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Question: {test_question}")
print(f"Generated Answer: {answer}")
