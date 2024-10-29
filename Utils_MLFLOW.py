# common_utils.py

import mlflow

def setup_mlflow_experiment(experiment_name):
    """Sets up MLflow with the specified experiment name and returns its ID."""
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Get or create the experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    return experiment_id