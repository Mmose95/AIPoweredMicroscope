import mlflow
from Utils_MLFLOW import setup_mlflow_experiment

experiment_id = setup_mlflow_experiment("Phase 1: Purulent Area")

def PH1_PurulentArea():
    metric = "PH1"

    with mlflow.start_run(experiment_id=experiment_id) as run:

        # Log parameters and metrics
        mlflow.log_param("test", metric)

    return metric

if __name__ == "__main__":
    PH1_PurulentArea()
