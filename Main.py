import mlflow

from Phase1.PH1_PurulentArea import PH1_PurulentArea
from Phase2.PH2_QualityAssessment import PH2_QualityAssessment
from Phase3.PH3_SpeciesDetermination import PH3_SpeciesDetermination

#renmeber to secure that the server is running:
#CLI --> mlflow server --host 127.0.0.1 --port 8080


########################## Phase 1: Purulent Area ##########################

PH1_PurulentArea()

########################## Phase 2: Quality Assessment ##########################

PH2_QualityAssessment()

########################## Phase 3: Species Determination ##########################

PH3_SpeciesDetermination()