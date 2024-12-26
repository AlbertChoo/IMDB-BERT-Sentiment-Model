import numpy as np
import tensorflow as tf

with open('movie_xids.npy', 'rb') as f:
  xids=np.load(f, allow_pickle=True)
with open('movie_xmasks.npy', 'rb') as f:
  xmasks=np.load(f, allow_pickle=True)
with open('labels.npy', 'rb') as f:
  labels=np.load(f, allow_pickle=True)

# Create TF dataset
dataset=tf.data.Dataset.from_tensor_slices((xids, xmasks, labels))
print("\nElement Spec after TF dataset: ", dataset.take(1))

# Rearrange dataset format
def map_func(input_ids, attention_masks, labels):
  return {'input_ids': input_ids, 'attention_mask': attention_masks}, labels

dataset=dataset.map(map_func)
print("\nElement Spec after rearrangement: ", dataset.take(1))

# Dataset Shuffling, Batch, Split and Save
BATCH_SIZE=8
dataset=dataset.shuffle(10000).batch(BATCH_SIZE, drop_remainder=True)

SPLIT=0.8
SIZE=int((xids.shape[0]/BATCH_SIZE)*SPLIT)
train_dataset=dataset.take(SIZE)
val_dataset=dataset.skip(SIZE)

# Save file, `use tf.data.Dataset.save(...) instead`
tf.data.experimental.save(train_dataset, 'train_dataset')
tf.data.experimental.save(val_dataset, 'val_dataset')
print("\nTrain and Val Dataset saved")

print("\nTrain Element Spec: ", train_dataset.element_spec)
print("\nVal Element Spec: ", val_dataset.element_spec)