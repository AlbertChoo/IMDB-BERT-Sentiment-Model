import tensorflow as tf
from transformers import TFAutoModel
import os
import zipfile

class BertSentimentModel(tf.keras.Model):
    def __init__(self, bert_model='bert-base-uncased', max_len=512):
        super().__init__()
        self.bert = TFAutoModel.from_pretrained(bert_model)
        self.dense1 = tf.keras.layers.Dense(1024, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid', name='outputs')
        
    def call(self, inputs):
        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        
        # Get BERT embeddings
        bert_outputs = self.bert(input_ids=input_ids, 
                               attention_mask=attention_mask,
                               return_dict=True)
        
        # Use the pooled output for classification
        pooled_output = bert_outputs.pooler_output
        
        # Dense layers
        x = self.dense1(pooled_output)
        return self.dense2(x)

# Create and compile model
def create_model():
    # Initialize model
    model = BertSentimentModel()
    
    # Freeze BERT layers
    model.bert.trainable = False
    
    # Compile model
    # Using standard Adam optimizer instead of legacy version
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    loss = tf.keras.losses.BinaryCrossentropy()
    metrics = [tf.keras.metrics.BinaryAccuracy('accuracy')]
    
    model.compile(optimizer=optimizer, 
                 loss=loss,
                 metrics=metrics)
    
    return model

# Load datasets
val_ds = tf.data.experimental.load('val_dataset')
train_ds = tf.data.experimental.load('train_dataset', element_spec=val_ds.element_spec)

# Create and compile model
model = create_model()

# Train model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2  # Adjust as needed
)

# Save the model
print("\nSaving Model...")
model.save('BERT-uncased-model.keras')

# # Optional: Download the Model to Your Local Machine, from Google Colab
# # Compress the model into a .zip file
# model_filename = 'BERT-uncased-model.keras'
# zip_filename = 'BERT-uncased-model.zip'

# with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
#     zipf.write(model_filename)

# print(f"Model saved and compressed as {zip_filename}")

# from google.colab import files

# files.download(zip_filename)