import tensorflow as tf
import numpy as np
from transformers import BertTokenizer

def predict_sentiment(model, text, tokenizer, seq_len=512):
    """
    Make sentiment predictions on new text input.
    Returns prediction label (positive/negative) instead of probabilities.
    
    Args:
        model: Trained BertSentimentModel instance
        text: String or list of strings to analyze
        tokenizer: BERT tokenizer
        seq_len: Maximum sequence length
    
    Returns:
        List of predictions ("positive" or "negative")
    """
    # Handle single string input
    if isinstance(text, str):
        text = [text]
    
    # Tokenize input text
    tokens = tokenizer(text,
                      max_length=seq_len,
                      padding=True,
                      truncation=True,
                      return_tensors='np',
                      add_special_tokens=True)
    
    # Create batch of size matching the input
    batch_size = len(text)
    dataset = tf.data.Dataset.from_tensor_slices((
        tokens['input_ids'], 
        tokens['attention_mask']
    )).batch(batch_size)
    
    # Format input and get predictions
    for batch in dataset:
        input_ids, attention_mask = batch
        model_input = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        predictions = model(model_input, training=False)
        
    # Convert probabilities to labels
    predictions = predictions.numpy()
    labels = ["positive" if pred >= 0.5 else "negative" for pred in predictions]
    
    return labels

def evaluate_model(model, dataset):
    """
    Evaluate model performance on a validation/test dataset.
    
    Args:
        model: Trained BertSentimentModel instance
        dataset: TensorFlow dataset in the format created earlier
        
    Returns:
        Dictionary containing metrics (accuracy, precision, recall, f1)
    """
    # Use model.evaluate() which is much faster than manual prediction
    results = model.evaluate(dataset, verbose=1)
    
    # Get metrics based on your model's compiled metrics
    metrics = {
        'loss': results[0],
        'accuracy': results[1]
    }
    
    # If you need additional metrics, you can compute them separately
    y_pred = []
    y_true = []
    
    # Use model() instead of predict() and process in batches
    for batch in dataset:
        x, y = batch
        pred = model(x, training=False)
        y_pred.extend(pred.numpy() >= 0.5)
        y_true.extend(y.numpy())
    
    # Convert to numpy arrays for metric calculation
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # Calculate additional metrics
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    metrics.update({
        'precision': precision,
        'recall': recall,
        'f1': f1
    })
    
    return metrics

# For making predictions:
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
example_text = ["This movie was fantastic!", "I really hated this film."]
predictions = predict_sentiment(model, example_text, tokenizer)

# # Example usage:
# """
# # For making predictions:
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# example_text = ["This movie was fantastic!", "I really hated this film."]
# predictions = predict_sentiment(model, example_text, tokenizer)

# # For evaluation:
# metrics = evaluate_model(model, val_ds)
# print("Validation Metrics:", metrics)
# """

# Print predictions
print("\nPredictions:")
for text, pred in zip(example_text, predictions):
    print(f"Text: '{text}'")
    print(f"Sentiment: {pred}\n")

# # For evaluation:
# print("Evaluating model on validation dataset...")
# metrics = evaluate_model(model, val_ds)
# print("\nValidation Metrics:")
# for metric_name, value in metrics.items():
#     print(f"{metric_name}: {value:.4f}")