import tensorflow as tf
import pandas as pd
import numpy as np
from transformers import BertTokenizer

# Check GPU availability
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Import Dataset
df=pd.read_csv("IMDB Dataset.csv")

seq_len=512
num_samples=len(df)

# 'np'=numpy,'tf'=tensorflow,'pt'=pytorch
tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
tokens=tokenizer(df['review'].tolist(), 
                 max_length=seq_len, 
                 padding=True, 
                 truncation=True, 
                 return_tensors='np', 
                 add_special_tokens=True)
print("Tokens keys: \n", tokens.keys())
print("Tokens shape: \n", tokens['input_ids'].shape)

print("Saving Tokens...")
# Save Tokens into Numpy Binary format
np.save('movie_xids.npy', tokens['input_ids'])
np.save('movie_xmasks.npy', tokens['attention_mask'])
np.save('movie_type_xids.npy', tokens['token_type_ids'])

# Save Memory
del tokens

print("Saving Labels after mapping...")
# One Hot Encoding labels
df['sentiment'].unique()
labels=np.array(df['sentiment'].map({'positive': 1, 'negative': 0}))
np.save('labels.npy', labels)

# # Test Removing HTML Tags, Handling Emojis, Removing Excess Whitespace
# df['review'] = df['review'].str.replace(r'<.*?>', '', regex=True)
# df['review'] = df['review'].apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))
# df['review'] = df['review'].str.strip().str.replace(r'\s+', ' ', regex=True)

def __init__(self):
    pass