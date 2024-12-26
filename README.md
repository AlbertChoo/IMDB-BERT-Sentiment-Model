Run `.py` file in sequence:
  1. data-preprocessing.py     # Data preprocessing, import data, etc
  2. dataset-input-pipeline.py # Building pipeline for preparing data to feed into model
  3. models.py                 # Build model, set-up arguments, then train it and save it
  4. make-predictions.py       # Make predictions on user's input, and evaluate val_dataset

A little guides on notebook:
  - Run `BERT_IMDB_sentiment_model` to test out whether computer is capable on running, and check the accuracy as well.

Notes: I did not run the code on my own local laptop due to OOM, I run it on Google Colab with 2 epochs, just to ensure it is working with well-performance, and it is done, fine-tuning those can be tried if u willing to, just to slowly unfreeze those layer until you're satisfied with the results.
Return tensors of `tf` instead of `np` on making predictions, due to different scenario. If you find out that my project is helping you, please make sure to STAR it!! Thanks.
