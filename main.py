import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 加载模型和分词器
model_path = 'models/t5_hotel_model.pth'
tokenizer = T5Tokenizer.from_pretrained('./models/mengzi-t5-base')  # 根据需要选择合适的模型
model = T5ForConditionalGeneration.from_pretrained('./models/mengzi-t5-base')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    model.cuda()
# 推理
model.eval()
# 加载模型参数
model.load_state_dict(torch.load("models/t5_hotel_model.pth"))  # 加载保存的模型参数
test_question = "可以洗衣服吗？"
input_text = f"Question: {test_question} Answer:"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

# 模型生成答案
output_ids = model.generate(input_ids, max_length=50)
answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Question: {test_question}")
print(f"Generated Answer: {answer}")
