import mlflow

from Preprocessing.DataHandling.DataLoader import loadImages
from Utils_MLFLOW import setup_mlflow_experiment

def qualityAssessment(trackExperiment, dataset):

    all_patches, all_labels = dataset

    experiment_id = setup_mlflow_experiment("Main Phase: Quality Assessment")

    metric = "PH2"

    OrgImages = loadImages("E:/PhdData/Original Data/Hvidovre/10x10")

    if trackExperiment == True:
        with mlflow.start_run(experiment_id=experiment_id) as run:

            # Log parameters and metrics
            mlflow.log_param("test", metric)


