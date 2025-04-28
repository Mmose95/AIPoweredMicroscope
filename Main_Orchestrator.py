import subprocess
import warnings
import logging
import multiprocessing
from MainPhase_QualityAssessment.MainPhase_QualityAssessment import qualityAssessment_SSL
import os

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
train_QualityAssessment_SSL = True
CreatePatchesFromFullSizeImages = False
train_QualityAssessment_Supervised = False
train_SpeciesDetermination = False


"""Initialize MLflow server for capturing experiments (remember to set "track experiment = true)"""
if __name__ == "__main__":
    subprocess.Popen(["StartMLflowServer.cmd"], shell=True)

########################## Phase 1: Quality Assessment ##########################

if train_QualityAssessment_SSL == True:

    '''training process'''
    if __name__ == "__main__":
        multiprocessing.freeze_support()  # Optional, safe to add
        qualityAssessment_SSL(True, "C:/Users/SH37YE/Desktop/FullSizeSamples/SSL_Training/TrainingPatches")

if train_QualityAssessment_Supervised == True:

    something = 1

stop = 1

########################## Phase 2: Species Determination ##########################

if train_SpeciesDetermination == True:
 bob = 1
