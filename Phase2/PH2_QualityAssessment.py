import mlflow

from DataHandling.DataLoader import loadImages
from Utils_MLFLOW import setup_mlflow_experiment

experiment_id = setup_mlflow_experiment("Phase 2: Quality Assessment")

def PH2_QualityAssessment():

    metric = "PH2"

    OrgImages = loadImages("E:/PhdData/Original Data/Hvidovre/10x10")


    with mlflow.start_run(experiment_id=experiment_id) as run:

        # Log parameters and metrics
        mlflow.log_param("test", metric)

    return metric

#if __name__ == "__main__":
    #PH2_QualityAssessment()

