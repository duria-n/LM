# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from datasets import load_dataset
# from transformers import Trainer, TrainingArguments
# import os



# model_name = 'bert-base-uncased'
# test_model = AutoModelForSequenceClassification.from_pretrained(model_name)
# test_tokenizer = AutoTokenizer.from_pretrained(model_name)

# train_dataset = load_dataset('imdb', split='train')
# test_dataset = load_dataset('imdb', split='test')

# curr_dir = os.path.dirname(os.path.abspath(__file__))
# output_dirs = os.path.join(curr_dir, r'result')
# # test_train_argument = TrainingArguments(output_dir=output_dirs,
# #                                         num_train_epochs=3,
# #                                         per_device_train_batch_size=8,
# #                                         evaluation_strategy='epoch',
# #                                         )

# # test_train_argument = TrainingArguments(
# #     output_dir=output_dirs,
# #     num_train_epochs=3,
# #     per_device_train_batch_size=16,
# #     learning_rate=2e-5,
# #     weight_decay=0.01,
# #     evaluation_strategy='epoch',
# # )
# test_train_argument = TrainingArguments(
#     output_dir=output_dirs,
#     num_train_epochs=3,
#     per_device_train_batch_size=16,
#     learning_rate=2e-5,
#     weight_decay=0.01,
#     evaluation_strategy='epoch',
#     save_strategy='steps',
#     fp16=True
# )


# trainer = Trainer(model=test_model,
#                   args=test_train_argument,
#                   train_dataset=train_dataset,
#                   eval_dataset=test_dataset,
#                   )


# trainer.train()
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import os

# 加载模型和分词器
model_name = 'bert-base-uncased'
test_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
test_tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载 IMDB 数据集的训练和测试分割
train_dataset = load_dataset('imdb', split='train')
test_dataset = load_dataset('imdb', split='test')

# 将文本数据转换为模型输入格式
def tokenize_function(example):
    return test_tokenizer(example['text'], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 设置输出文件夹路径
curr_dir = os.path.dirname(os.path.abspath(__file__))
output_dirs = os.path.join(curr_dir, 'result')

# 设置训练参数
training_args = TrainingArguments(
    output_dir=output_dirs,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir=os.path.join(output_dirs, 'logs'),
    load_best_model_at_end=True,
)

# 初始化 Trainer
trainer = Trainer(
    model=test_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# 微调模型
trainer.train()
