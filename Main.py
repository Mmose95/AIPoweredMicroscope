import subprocess
import tensorflow as tf
import numpy as np
from DataHandling.DataLoader import loadImages
from numpy import load

from Preprocessing.dataAugmentation import dataAugmentation

##Switches/Modes for training different networks.
train_ph1 = False
train_ph2 = True
train_ph3 = False

#Load all training, validation and test data. (already patched from earlier) - Both input and output is in their respective 'xxx_data'
all_data = load('C:/Users/mose_/AAU/Sundhedstek/PhD/AIPoweredMicroscope_development/DataHandling/InputPatches_Hvidovre.npz')
#validation_data = load('C:/Users/mose_/AAU/Sundhedstek/PhD/AIPoweredMicroscope_development/DataHandling/InputPatches_Hvidovre.npz')
#test_data = load('C:/Users/mose_/AAU/Sundhedstek/PhD/AIPoweredMicroscope_development/DataHandling/InputPatches_Hvidovre.npz')

#Convert to ndarrays
all_data_input = all_data['arr_0']
all_data_output = all_data['arr_1']


#Data augmentation (flip, color jittering, etc)

all_data_inpit_aug = dataAugmentation()

#Normalization (for training purposes)

#renmeber to secure that the server is running:
#CLI --> mlflow server --host 127.0.0.1 --port 5000

# Replace 'path_to_your_cmd_file.cmd' with the actual path to your .cmd file
subprocess.Popen(["StartMLflowServer.cmd"], shell=True)

########################## Phase 1: Purulent Area ##########################
if train_ph1 == True:
    from Phase1.PH1_PurulentArea import PH1_PurulentArea


    PH1_PurulentArea()

########################## Phase 2: Quality Assessment ##########################
if train_ph2 == True:
    from Phase2.PH2_QualityAssessment import PH2_QualityAssessment

    loadImages()


    PH2_QualityAssessment()

########################## Phase 3: Species Determination ##########################

if train_ph3 == True:
    from Phase3.PH3_SpeciesDetermination import PH3_SpeciesDetermination


    PH3_SpeciesDetermination()

