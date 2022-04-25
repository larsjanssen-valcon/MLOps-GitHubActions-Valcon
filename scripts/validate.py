# Databricks notebook source
from mlflow.tracking import MlflowClient

client = MlflowClient()

model_name = "classifier"

# COMMAND ----------

def get_models(model_name):
    all_models = [model for model in client.search_model_versions(f"name='{model_name}'")]
    all_models = sorted(all_models, key= lambda x: x.version, reverse=True)
    production_models = [model for model in all_models if model.current_stage=="Production"]
    if len(production_models) == 0:
      return all_models[0], None
    elif len(all_models) == 0:
      raise ValueError("No models found")
    elif len(production_models) > 1:
      raise ValueError("Cannot resolve registered production models to a single model")
    else:
      return all_models[0], production_models[0]

def get_accuracy_model(model):
    return client.get_run(model.run_id).data.metrics["acc"]

# COMMAND ----------

last_model, production_model = get_models(model_name)

if not production_model:
  print(f"Transitioning last {model_name} model to production since there is no model in production yet.")
  client.transition_model_version_stage(
    name=model_name,
    version=last_model.version,
    stage="Production"
  )
  dbutils.notebook.exit("Model successfully transitioned to production, exiting...")

acc_last = get_accuracy_model(last_model)
acc_prod = get_accuracy_model(production_model)
print(f"Production {model_name} model version {production_model.version} accuracy: {acc_prod}")
print(f"Latest {model_name} model version {last_model.version} accuracy: {acc_last}")

# COMMAND ----------

if acc_last > acc_prod:
    print(f"Archiving {model_name} model version {production_model.version} ")
    client.transition_model_version_stage(
    name=model_name,
    version=production_model.version,
    stage="Archived"
)
    print(f"Transitioning production {model_name} model version from {production_model.version} to {last_model.version}")
    client.transition_model_version_stage(
    name=model_name,
    version=last_model.version,
    stage="Production"
)
else:
    print(f"Not transitioning new {model_name} model to production since most accurate version remains {production_model.version}")
