from transformers import AutoModelForSequenceClassification
from transformers import pipeline

#加载模型指定标签数量
model = AutoModelForSequenceClassification.from_pretrained('./model',num_labels=2)

classifier = pipeline('sentiment-analysis',model="./cache/checkpoint-500")
sequence = """nectarine"""
res =  classifier(sequence)
print(res)