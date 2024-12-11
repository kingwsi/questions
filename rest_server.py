import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from flask import Flask, request, jsonify, Response
import json

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
model.load_state_dict(torch.load(model_path))  # 加载保存的模型参数

app = Flask(__name__)

@app.route('/infer', methods=['GET'])
def infer():
    question = request.args.get('question')
    # ... 处理问题并生成推理结果 ...
    input_ids = tokenizer(question, return_tensors="pt").input_ids.to(device)
    # 模型生成答案
    output_ids = model.generate(input_ids, max_length=50)
    # 检查生成的答案是否有效
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"Question: {question}")
    print(f"Generated Answer: {answer}")

    # 手动构造 JSON 响应，使用 json.dumps() 并设置 ensure_ascii=False
    response_data = json.dumps({'answer': answer}, ensure_ascii=False)

    # 返回自定义响应
    return Response(response_data, content_type='application/json; charset=utf-8')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8899)
