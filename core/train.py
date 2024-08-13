from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
import pandas as pd
import datasets
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers.trainer import IntervalStrategy

# 训练数据集
train_data = "./data/custom_fruit.csv"
# 验证数据集
valid_data = "./data/custom_fruit.csv"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("./model")
print(tokenizer)

# 读取数据
df = pd.read_csv(train_data)
ds = datasets.Dataset.from_pandas(df)
# 乱序
ds = ds.shuffle(42)
print(ds[0])

# 分词
tokens = tokenizer.tokenize(ds["text"][0])
print("tokens=", tokens)

# 编码
encode = tokenizer.encode(ds["text"][0])
print("encode=", encode)

# 解码
decode = tokenizer.decode(encode)
print("decode=", decode)

# 词转编码
ids = tokenizer.convert_tokens_to_ids(tokens)
print("ids = ", ids)

# 加载模型指定标签数量
model = AutoModelForSequenceClassification.from_pretrained('./model',num_labels=20)
# print(model)

# 数据dataset
data_files = {"train": train_data, "test": valid_data}
raw_datasets = datasets.load_dataset("csv", data_files=data_files, delimiter=",")

print(raw_datasets['train'][0])

def tokenize_function(sample):
    return tokenizer(sample['text'], max_length=300, truncation=True)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# 计算指标
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# 训练参数
training_args = TrainingArguments(output_dir='./cache', # 指定输出文件夹，没有会自动创建
                                evaluation_strategy="epoch",
                                per_device_train_batch_size=32,
                                per_device_eval_batch_size=32,
                                learning_rate=5e-5,
                                warmup_ratio=0.2,
                                logging_dir='./logs',
                                logging_steps=100,
                                max_steps=900,
                                save_steps=300,
                                report_to="tensorboard")
# 训练器
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics)

# 开始训练
trainer.train()
