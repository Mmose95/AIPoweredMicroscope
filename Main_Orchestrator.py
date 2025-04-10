import subprocess

from Preprocessing.DataHandling.DataLoader import loadImages
from numpy import load
from Preprocessing.dataAugmentation import dataAugmentation
from MainPhase_QualityAssessment.MainPhase_QualityAssessment import qualityAssessment_supervised, qualityAssessment_SSL
from Preprocessing.preprocessing_QA_Main import preprocessing_QA_Supervised

##Switches/Modes for training different networks.
train_QualityAssessment = True
train_SpeciesDetermination = False


"""Initialize MLflow server for capturing experiments (remember to set "track experiment = true)"""

subprocess.Popen(["StartMLflowServer.cmd"], shell=True)

########################## Phase 2: Quality Assessment ##########################

if train_QualityAssessment == True:
    '''preprocessing'''

    dataset_SSL = preprocessing_QA_SSL(CreateOrLoadPatches='load', vizLabels=True)
    dataset_supervised = preprocessing_QA_Supervised(CreateOrLoadPatches='load', vizLabels=True)

    '''training process'''

    qualityAssessment_SSL(trackExperiment=True, dataset=dataset)

    qualityAssessment_supervised(trackExperiment=True, dataset=dataset)

stop = 1

########################## Phase 3: Species Determination ##########################

if train_SpeciesDetermination == True:
 bob = 1

##''' "C:\Users\mose_>code2flow X:\AAU\Sundhedstek\PhD\AIPoweredMicroscope_development -o diagram.png '''