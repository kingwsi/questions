from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import AdamW

# 加载 T5 模型
model_name = "langboat/mengzi-t5-base"  # 可以尝试 t5-small 或 t5-large
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 数据集
data = [
    {"question": "酒店附近有哪些景点", "answer": "酒店周边有动物园，以及著名5A 景区，坐地铁 2 站就到达"},
    {"question": "有儿童乐园吗", "answer": "酒店免费提供儿童亲子乐园，不限时长可免费游玩"},
    {"question": "可以洗衣服吗", "answer": "可以免费洗衣以及烘干"},
    {"question": "早餐是几点开始", "answer": "早餐时间为早上 6:00 到 8:30 分"},
    {"question": "送早餐吗", "answer": "部分房型含免费早餐，具体可咨询前台"},
    {"question": "有洗衣房吗", "answer": "二楼设有免费洗衣房，可以洗衣服以烘干及熨烫"},
    # 添加更多数据
]

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

# 优化器
optimizer = AdamW(model.parameters(), lr=5e-5)

# 训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()

for epoch in range(10):  # 增加训练轮数
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

# 推理
model.eval()
test_question = "附近景点"
input_text = f"Question: {test_question} Answer:"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

# 模型生成答案
output_ids = model.generate(input_ids, max_length=50)
answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Question: {test_question}")
print(f"Generated Answer: {answer}")
