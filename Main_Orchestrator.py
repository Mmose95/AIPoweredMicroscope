import subprocess
import warnings
import logging
import multiprocessing
import os

from Helpers_General.Supervised_learning_helpers.dino_backbone_for_yolo import DINOv2Backbone
from MainPhase_QualityAssessment.Main_QualityAssessment_SSL_DINOV2 import qualityAssessment_SSL_DINOV2
from MainPhase_QualityAssessment.Main_QualityAssessment_Supervised_YOLO import qualityAssessment_supervised_YOLO

globals()["Helpers_General.Supervised_learning_helpers.dino_backbone_for_yolo.DINOv2Backbone"] = DINOv2Backbone

# Suppress irrelevant warnings
warnings.filterwarnings("ignore", message=".*xFormers is available.*")
warnings.filterwarnings("ignore", message=".*No module named 'triton'.*")
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")
logging.getLogger().setLevel(logging.ERROR)

# Suppress all warnings (recommended for production-like runs)
warnings.filterwarnings("ignore")

# Hide Python warnings emitted from imported libraries
os.environ["PYTHONWARNINGS"] = "ignore"

# Reduce logging from other packages (like timm or xFormers)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("xformers").setLevel(logging.ERROR)
logging.getLogger("dinov2").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

##Switches/Modes for training different networks.
train_QualityAssessment_SSL = False
train_QualityAssessment_Supervised = True
train_SpeciesDetermination = False


"""Initialize MLflow server for capturing experiments (remember to set "track experiment = true)"""
if __name__ == "__main__":
    subprocess.Popen(["StartMLflowServer.cmd"], shell=True)

########################## Phase 1: Quality Assessment ##########################


''' Prerequisite for SSL training:

Create (unlabeled) patches for training of selfsupervised model:
use Preprocessing > DataHandling > PatchCreationSSL.py to generate the patches needed
'''

if train_QualityAssessment_SSL == True:

    '''training process - Self-Supervised'''
    if __name__ == "__main__":
        multiprocessing.freeze_support()  # Optional, safe to add
        qualityAssessment_SSL_DINOV2(True, "C:/Users/SH37YE/Desktop/FullSizeSamples/SSL_Training/TrainingPatches")

''' Prerequisite for supervised training:

Create labeled data for training of supervised model:
use Helpers_General > Supervised_learning_helpers > MainAreaSelection to generate candidate regions. 
Upload these to CVAT via treat server to be labeled by Herlev (or other experts) 
Download the labeled data from Cvat and convert the labels to fit supervised training (patching and label conversion)
'''

if train_QualityAssessment_Supervised == True:

    SSL_encoder_name = "./Checkpoints/" + "ExId_854681636342556727_run_20250428_154510_BEST_dinov2_selfsup_trained.pt"

    '''Training process - Supervised'''
    import ultralytics.nn.tasks as yolo_tasks
    from Helpers_General.Supervised_learning_helpers.dino_backbone_for_yolo import DINOv2Backbone

    # 👇 Correct injection
    yolo_tasks.__dict__[
        "Helpers_General.Supervised_learning_helpers.dino_backbone_for_yolo.DINOv2Backbone"] = DINOv2Backbone

    # Now call your training logic
    qualityAssessment_supervised_YOLO(False, SSL_encoder_name)

    stop = 1

########################## Phase 2: Species Determination ##########################

if train_SpeciesDetermination == True:
 bob = 1
