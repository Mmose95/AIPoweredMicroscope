import subprocess
import multiprocessing

from Helpers_General.FolderSelection_input import select_training_folder, select_encoder_file
from Helpers_General.FullDataSplits import fullDataSplits
from Helpers_General.FullDataSplitsSafe import fullDataSplits_SampleSafe
from Helpers_General.Supervised_learning_helpers.Cloned_DINOV2_backbone_for_yolo import DinoV2New
from Helpers_General.Supervised_learning_helpers.lambda_helper import TupleSelect
from MainPhase_QualityAssessment.Main_QualityAssessment_SSL_DINOV2 import qualityAssessment_SSL_DINOV2
from MainPhase_QualityAssessment.Main_QualityAssessment_Supervised_DINO import qualityAssessment_supervised_DINO
from MainPhase_QualityAssessment.Main_QualityAssessment_Supervised_RFDETR import qualityAssessment_supervised_RFDETR


def ask_switch(question):
    response = input(f"{question} (y/n): ").strip().lower()
    return response == 'y'

# Top-level switches - hardcoding
'''generate_overall_dataSplits = False
train_QualityAssessment_SSL = True
train_QualityAssessment_Supervised = False
train_SpeciesDetermination = False'''


def main():

    # Top-level switches - dynamically.
    SSL_Training_Data_Path = None
    Supervised_Training_Data_Path = None

    # Interactive switches
    generate_overall_dataSplits = ask_switch("Generate overall data splits?")

    train_QualityAssessment_SSL = ask_switch("Train Quality Assessment (Self-Supervised)?")

    if train_QualityAssessment_SSL == True:
        train_SSL_Sup_in_Conjunction = ask_switch("Train SSL and Supervised in conjunction? (i.e. automatically use the model trained in SSL to continue training supervised")

        if train_SSL_Sup_in_Conjunction == True:
            train_QualityAssessment_Supervised = True
            print(f"Conjunction training selected. Selecting encoder from SSL phase automatically.")

            SSL_Training_Data_Path = select_training_folder("Select data for SELF-SUPERVISED training")
            Supervised_Training_Data_Path = select_training_folder("Select data for SUPERVISED training")

        else:
            SSL_Training_Data_Path = select_training_folder("Select data for SELF-SUPERVISED training")
    else:
        train_QualityAssessment_Supervised = ask_switch("Train Quality Assessment (Supervised)?")
        if train_QualityAssessment_Supervised:
            print(f"To train a supervised model, we need a pretrained encoder as backbone")
            SSL_encoder_name = select_encoder_file()
            Supervised_Training_Data_Path = select_training_folder("Select data for SUPERVISED training")

    train_SpeciesDetermination = ask_switch("Train Species Determination?")

    # Start MLflow server
    subprocess.Popen(["StartMLflowServer.cmd"], shell=True)

    ''''# Prompt for data
    if train_QualityAssessment_SSL:
        print("Select the folder containing the training data for the SSL phase.")
        SSL_Training_Data_Path = select_training_folder()

    if train_QualityAssessment_Supervised:
        print("Select the folder containing data for the Supervised phase.")
        Supervised_Training_Data_Path = select_training_folder()'''


    # Generate data splits
    if generate_overall_dataSplits:
        fullDataSplits_SampleSafe(
            r"C:\Users\SH37YE\Desktop\Clinical_Bacteria_DataSet\DetectionDataSet\images",
            r"C:\Users\SH37YE\Desktop\Clinical_Bacteria_DataSet\DetectionDataSet\labels",
            r"C:\Users\SH37YE\Desktop\Clinical_Bacteria_DataSet\DetectionDataSet"
        )

    ####################################### Phase 1: SSL #######################################
    if train_QualityAssessment_SSL:
        if SSL_Training_Data_Path:
            SSL_encoder_name = qualityAssessment_SSL_DINOV2(True, SSL_Training_Data_Path)
        else:
            print("❌ No SSL data folder selected. Aborting.")

    ####################################### Phase 2: Supervised #######################################
    if train_QualityAssessment_Supervised:

        qualityAssessment_supervised_RFDETR(
            trackExperiment=True,
            encoder_name=SSL_encoder_name,
            supervised_data_path = Supervised_Training_Data_Path,
        )

        ''' DINO REPO 
        if Supervised_Training_Data_Path:
            qualityAssessment_supervised_DINO(
                trackExperiment=False,
                encoder_name=SSL_encoder_name,
                train_dataset=Supervised_Training_Data_Path,
                val_dataset=Supervised_Validation_Data_Path,
            )
        else:
            print("❌ No Supervised training folder selected. Aborting.")
            '''

    if train_SpeciesDetermination:
        print("Phase 3 logic not implemented yet.")

# Required to prevent multiprocessing issues on Windows
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
