from sentence_transformers import SentenceTransformer, util
import torch
# 加载 BERT 的句子嵌入模型
model = SentenceTransformer('hfl/chinese-bert-wwm')

# 数据示例
answers = [
    "酒店周边有动物园，以及著名5A 景区，坐地铁 2 站就到达",
    "酒店免费提供儿童亲子乐园，不限时长可免费游玩",
    "可以免费洗衣以及烘干",
    "早餐时间为早上 6:00 到 8:30 分",
    "部分房型含免费早餐，具体可咨询前台",
    "二楼设有免费洗衣房，可以洗衣服以烘干及熨烫"
]
questions = [
    "酒店附近有哪些景点?",
    "有儿童乐园吗?",
    "可以洗衣服吗?",
    "早餐是几点开始?",
    "送早餐吗?",
    "有洗衣房吗?",
]

# 计算答案和问题的向量
answer_embeddings = model.encode(answers)
question_embeddings = model.encode(questions)

# 新问题的向量
new_question = "附近有什么好玩的？"
new_question_embedding = model.encode(new_question)

# 计算相似度
similarities = util.pytorch_cos_sim(new_question_embedding, answer_embeddings)
print("相似度:", similarities)
best_match_idx = torch.argmax(similarities)
best_answer = answers[best_match_idx]

print(f"Question: {new_question}")
print(f"Best Answer: {best_answer}")
