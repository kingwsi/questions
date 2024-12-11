from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import AdamW

# 加载 T5 模型
model_name = "./models/mengzi-t5-base"  # 可以尝试 t5-small 或 t5-large
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 数据集
import json

with open('QA.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 数据预处理
inputs = [f"Question: {item['question']} Answer:" for item in data]
labels = [item['answer'] for item in data]

encoded_inputs = tokenizer(inputs, max_length=128, padding=True, truncation=True, return_tensors="pt")
encoded_labels = tokenizer(labels, max_length=128, padding=True, truncation=True, return_tensors="pt")
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

# 训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Starting train by {device}")
model.to(device)
model.train()

for epoch in range(20):  # 增加训练轮数
    total_loss = 0
    for batch in dataloader:
        batch = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

# 训练结束后保存模型
torch.save(model.state_dict(), "models/t5_hotel_model.pth")  # 保存模型参数

# 推理
model.eval()
# 加载模型参数
model.load_state_dict(torch.load("models/t5_hotel_model.pth"))  # 加载保存的模型参数
test_question = "附近有什么景点"
input_text = f"Question: {test_question} Answer:"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

# 模型生成答案
output_ids = model.generate(input_ids, max_length=50)
answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Question: {test_question}")
print(f"Generated Answer: {answer}")
