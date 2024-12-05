import mlflow
from Utils_MLFLOW import setup_mlflow_experiment


def speciesDetermination(trackExperiment):

    experiment_id = setup_mlflow_experiment("Phase 3: Species Determination")

    metric = "PH3"

    if trackExperiment == True:
        with mlflow.start_run(experiment_id=experiment_id) as run:

            # Log parameters and metrics
            mlflow.log_param("test", metric)

