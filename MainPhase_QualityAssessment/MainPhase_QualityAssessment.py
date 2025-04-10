import mlflow
import torch
import torch
import timm
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from Helpers_General.Self_Supervised_learning_Helpers.SSL_functions import PILImageDataset, MicroscopyDataset, \
    ssl_transform
from MainPhase_QualityAssessment.Main_Pretext_SSL_DINOYOLO_Training import DINO_training_loop
from Utils_MLFLOW import setup_mlflow_experiment
from ultralytics import YOLO

def qualityAssessment_SSL(trackExperiment_QualityAssessment_SSL, trackExperiment_QualityAssessment_Supervised):

    #Setting up mlflow experimentation
    experiment_id = setup_mlflow_experiment("Main Phase: Quality Assessment (SelfSupervised)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ''''''''''''''''''''''''''''''''''''''' SSL Pretext '''''''''''''''''''''''''''''''''''
    ''' The starting point of this SSL pretext training is an already pretrained model: using the dino model'''

    #SSL parameters
    ssl_data_path = "D:\PHD\PhdData\SSL_DATA_PATCHESTest"
    pretrainedmodel_name = "vit_small_patch14_dinov2.lvd142m"
    batch_size = 16
    learning_rate = 1e-4
    weight_decay=1e-5
    workers = 4
    num_epochs = 1

    model = timm.create_model(model_name=pretrainedmodel_name, pretrained=True)
    dataset = MicroscopyDataset(ssl_data_path, transform=ssl_transform)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers= workers)
    optimizer = optim.AdamW(model.parameters(), learning_rate, weight_decay= weight_decay)
    lossFunction = nn.MSELoss() # Loss: MSE Loss for simplicity ---> Need to change this to the real DINO loss when actually training for results!

    DINO_training_loop(model, optimizer, lossFunction, num_epochs, dataloader, device)

    metricSSL = "PH1_SSL"


    if trackExperiment_QualityAssessment_SSL == True:
        with mlflow.start_run(experiment_id=experiment_id) as run:

            # Log parameters and metrics
            mlflow.log_param("SSL_METRIC_TEST", metricSSL)


    ''''''''''''''''''''''''' Downstream task (Bounding box Classification)'''''''''''''''''''''''''''''''''

    ### Setting up mlflow experimentation
    experiment_id = setup_mlflow_experiment("Main Phase: Quality Assessment (Supervised)")

    ### Full model architecture configuration.

    # Load a YOLOv8 model
    yolo_model = YOLO("yolov8s.pt")  # Load YOLOv8 Small for faster training

    # Load pretrained DINOv2
    dinov2_model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False)
    dinov2_model.load_state_dict(torch.load("dinov2_pretrained_microscopy.pth"))

    # Remove DINOv2's classification head and use it as a feature extractor
    dinov2_model.head = torch.nn.Identity()

    # Assign DINOv2 as YOLO’s new backbone
    yolo_model.model.model[0] = dinov2_model

    # Save new model config
    yolo_model.save("yolov8_with_dino.pt") #This is the actual full model architechture (YoloV8 model with (pretrained) DINOV2 backbone)

    ### Fine-tuning of full model architecture (using labeled data)


    metricSupervised = "PH1_Supervised"

    if trackExperiment_QualityAssessment_Supervised == True:
        with mlflow.start_run(experiment_id=experiment_id) as run:
            # Log parameters and metrics
            mlflow.log_param("SUPERVISED_METRIC_TEST", metricSupervised)