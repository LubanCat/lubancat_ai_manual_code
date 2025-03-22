from transformers import BertTokenizer, BertModel

# 加载预训练模型和分词器，可自定义路径
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 输入处理
inputs = tokenizer("Hello, BERT!", return_tensors="pt")

# 前向传播
outputs = model(**inputs, output_hidden_states=True)

# 输出最后
last_hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)