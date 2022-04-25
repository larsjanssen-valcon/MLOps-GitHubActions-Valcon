# Databricks notebook source
# MAGIC %md
# MAGIC # Train Notebook
# MAGIC 
# MAGIC ## As a data scientist, this is where I write my training code.

# COMMAND ----------

# Import necessary modules
import argparse
import urllib
import os
import numpy as np
import pandas as pd
import mlflow
from mlflow.exceptions import RestException

# COMMAND ----------

# MAGIC %md
# MAGIC ### Give model a name

# COMMAND ----------

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# COMMAND ----------

dbutils.widgets.dropdown("train_type", "local", ["local", "register_model"], "Train type")
train_type = dbutils.widgets.get("train_type")

# COMMAND ----------

model_name = "classifier"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define datasets

# COMMAND ----------

# this is the URL to the CSV file containing the connected car component descriptions
cardata_url = ('https://quickstartsws9073123377.blob.core.windows.net/'
               'azureml-blobstore-0d1c4218-a5f9-418b-bf55-902b65277b85/'
               'quickstarts/connected-car-data/connected-car_components.csv')

cardata_ds_name = 'connected_car_components'
cardata_ds_description = 'Connected car components data'

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define dataset parameters

# COMMAND ----------

embedding_dim = 100
training_samples = 90000
validation_samples = 5000
max_words = 10000

# COMMAND ----------

# MAGIC %md
# MAGIC ### Process dataset

# COMMAND ----------

print('Downloading connected car components dataset...')
# Download the connected car components dataset

car_components_df = pd.read_csv(cardata_url)

print('Download complete.')

components = car_components_df["text"].tolist()
labels = car_components_df["label"].tolist()

print("Processing car components data completed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use the Tokenizer from Keras to "learn" a vocabulary from the dataset

# COMMAND ----------

print("Tokenizing data...")

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(components)
sequences = tokenizer.texts_to_sequences(components)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=embedding_dim)

labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
print("Tokenizing data complete.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create training, validation, and testing datasets

# COMMAND ----------

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]

x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

x_test = data[training_samples + validation_samples:]
y_test = labels[training_samples + validation_samples:]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build and train the model

# COMMAND ----------
workspace_dir = "/".join(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().split("/")[:-2]) # Folder of workspace

experiment_path = f"{workspace_dir}/{model_name}"
try:
    mlflow.set_experiment(experiment_path)
except RestException:
    # If experiment does not yet exist -> create experiment
    print("Experiment path does not yet exist; will create it now.")

    # For now set the experiment name equal to the given model_name
    mlflow.create_experiment(experiment_path)
    mlflow.set_experiment(experiment_path)

print("Saving experiment to: {}".format(experiment_path))

print("Training model...")
with mlflow.start_run():
    # Set model parameters
    params = {
        'learning_rate': 0.0001,
        'max_depth': 5,
        'n_estimators': 30
    }

    model = xgb.XGBClassifier(objective="binary:logistic", use_label_encoder=False, verbosity=0, **params)

    model.fit(x_train, y_train)

    print("Training model completed.")

    if train_type == "register_model":
        mlflow.xgboost.log_model(model, artifact_path="model", registered_model_name=model_name)

        print("Saving model in MLFlow...")

    for key in params.keys():
        mlflow.log_param(key, params[key])

    print("Saving model completed.")

    print("Saving model metrics...")

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric('acc', accuracy)

    print("Saving model metrics completed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### This is where the train script ends
