from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

# Create an MLflow Client
client = MlflowClient()

# List all deleted experiments
deleted_experiments = client.search_experiments(view_type=ViewType.DELETED_ONLY)

# Print the ID and name of each deleted experiment
for experiment in deleted_experiments:
    print(f"Experiment ID: {experiment.experiment_id}, Name: {experiment.name}")

# Replace with the ID of the experiment you want to restore
experiment_id_to_restore = "819762681029603408"

# Restore the experiment
client.restore_experiment(experiment_id_to_restore)
print(f"Restored experiment with ID: {experiment_id_to_restore}")