import warnings
import logging
import subprocess
import multiprocessing
import os
from types import SimpleNamespace

from Helpers_General.FolderSelection_input import select_training_folder
from Helpers_General.FullDataSplits import fullDataSplits
from Helpers_General.Supervised_learning_helpers.Cloned_DINOV2_backbone_for_yolo import DinoV2New
from Helpers_General.Supervised_learning_helpers.lambda_helper import TupleSelect
from MainPhase_QualityAssessment.Main_QualityAssessment_SSL_DINOV2 import qualityAssessment_SSL_DINOV2
from MainPhase_QualityAssessment.Main_QualityAssessment_Supervised_DINO import qualityAssessment_supervised_DINO

# Suppress irrelevant warnings
warnings.filterwarnings("ignore", message=".*xFormers is available.*")
warnings.filterwarnings("ignore", message=".*No module named 'triton'.*")
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("xformers").setLevel(logging.ERROR)
logging.getLogger("dinov2").setLevel(logging.ERROR)

# Top-level switches
generate_overall_dataSplits = False
train_QualityAssessment_SSL = False
train_QualityAssessment_Supervised = True
train_SpeciesDetermination = False

# Wrap the full program in main()
def main():
    # Start MLflow server
    subprocess.Popen(["StartMLflowServer.cmd"], shell=True)

    SSL_Training_Data_Path = None
    Supervised_Training_Data_Path = None
    Supervised_Validation_Data_Path = None

    # Prompt for data
    if train_QualityAssessment_SSL:
        print("Select the folder containing the training data for the SSL phase.")
        SSL_Training_Data_Path = select_training_folder()

    if train_QualityAssessment_Supervised:
        print("Select the folder containing the training data for the Supervised phase.")
        Supervised_Training_Data_Path = select_training_folder()
        print("Select the folder containing the validation data for the Supervised phase.")
        Supervised_Validation_Data_Path = select_training_folder()

    # Generate data splits
    if generate_overall_dataSplits:
        fullDataSplits(
            r"C:\Users\SH37YE\Desktop\Clinical Bacteria DataSet\DetectionDataSet\images",
            r"C:\Users\SH37YE\Desktop\Clinical Bacteria DataSet\DetectionDataSet\labels",
            r"C:\Users\SH37YE\Desktop\Clinical Bacteria DataSet\DetectionDataSet"
        )

    # Phase 1: SSL
    if train_QualityAssessment_SSL:
        if SSL_Training_Data_Path:
            qualityAssessment_SSL_DINOV2(True, SSL_Training_Data_Path)
        else:
            print("❌ No SSL data folder selected. Aborting.")

    # Phase 2: Supervised
    if train_QualityAssessment_Supervised:
        SSL_encoder_name = "./Checkpoints/ExId_854681636342556727_run_20250428_154510_BEST_dinov2_selfsup_trained.pt"
        if Supervised_Training_Data_Path:
            qualityAssessment_supervised_DINO(
                trackExperiment=False,
                encoder_name=SSL_encoder_name,
                train_dataset=Supervised_Training_Data_Path,
                val_dataset=Supervised_Validation_Data_Path,
            )
        else:
            print("❌ No Supervised training folder selected. Aborting.")

    if train_SpeciesDetermination:
        print("Phase 3 logic not implemented yet.")

# Required to prevent multiprocessing issues on Windows
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
