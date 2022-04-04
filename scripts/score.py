# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC #### Set-up

# COMMAND ----------

import os

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from mlflow.tracking import MlflowClient
from numpy.random import default_rng

# COMMAND ----------

MODEL_NAME = "classifier"
MODEL_STAGE = "Production"

EMBEDDING_DIM = 100
TRAINING_SAMPLES = 90000
VALIDATION_SAMPLES = 5000
MAX_WORDS = 10000

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Load model
# MAGIC 
# MAGIC Load the latest model from the MLflow model registry.

# COMMAND ----------

try:
    # Don't reload model to save time
    loaded_model
except NameError:
    # Load trained model from MLflow registry
    loaded_model = mlflow.xgboost.load_model(
        model_uri=os.path.join("models:", MODEL_NAME, MODEL_STAGE)
    )

# COMMAND ----------

print(loaded_model)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Load data
# MAGIC 
# MAGIC Retrieve data to run the trained model on. 

# COMMAND ----------

# This is the URL to the CSV file containing the connected car component descriptions
cardata_url = (
    "https://quickstartsws9073123377.blob.core.windows.net/"
    "azureml-blobstore-0d1c4218-a5f9-418b-bf55-902b65277b85/"
    "quickstarts/connected-car-data/connected-car_components.csv"
)

cardata_ds_name = "connected_car_components"
cardata_ds_description = "Connected car components data"

# COMMAND ----------

print("Downloading connected car components dataset...")
# Download the connected car components dataset

car_components_df = pd.read_csv(cardata_url)

print("Download complete.")

components = car_components_df["text"].tolist()
labels = car_components_df["label"].tolist()

print("Processing car components data completed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Use the Tokenizer from Keras to "learn" a vocabulary from the dataset

# COMMAND ----------

print("Tokenizing data...")

tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(components)
sequences = tokenizer.texts_to_sequences(components)

word_index = tokenizer.word_index
print("Found %s unique tokens." % len(word_index))

data = pad_sequences(sequences, maxlen=EMBEDDING_DIM)

labels = np.asarray(labels)
print("Shape of data tensor:", data.shape)
print("Shape of label tensor:", labels.shape)
print("Tokenizing data complete.")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create training, validation, and testing datasets

# COMMAND ----------

indices = np.arange(data.shape[0])
rng = default_rng(12345)  # set seed to reproduce results
rng.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_test = data[TRAINING_SAMPLES + VALIDATION_SAMPLES :]
y_test = labels[TRAINING_SAMPLES + VALIDATION_SAMPLES :]

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Make predictions
# MAGIC 
# MAGIC Apply the model to the test features. 

# COMMAND ----------

loaded_model.predict(xgb.DMatrix(x_test))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Example in tutorial
# MAGIC 
# MAGIC Run the model on a single sample. 

# COMMAND ----------

arr = np.array(
    [
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            8,
            2,
            5,
            6,
            4,
            3,
            1,
            34,
        ]
    ]
)

loaded_model.predict(xgb.DMatrix(arr))
