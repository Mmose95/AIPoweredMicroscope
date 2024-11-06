import subprocess

##Switches/Modes for training different networks.
train_ph1 = False
train_ph2 = True
train_ph3 = False


########################### Preprocessing ##########################
#Content: Load (already) pateched images,




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
    PH2_QualityAssessment()

########################## Phase 3: Species Determination ##########################

if train_ph3 == True:
    from Phase3.PH3_SpeciesDetermination import PH3_SpeciesDetermination
    PH3_SpeciesDetermination()

