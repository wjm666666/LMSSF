# LMSFF


## Dataset
   Download the CK+ dataset and RAVDESS and store them in data/dataset/*
    
    dataset/
            CK/
                - anger
                - contempt
                - disgust
                - fear
                - happy
                - sadness
                - suiprise
    
            RAVDESS/
                - Actor_01
                - Actor_02
                   ......
                - Actor_23
                - Actor_24
## Overview
### Overall Architecture for LMSFF
#### <img width="525" alt="4424031a99009a43f6d2b60b8b9405b" src="https://github.com/wjm666666/LMSSF/assets/60913990/f3ecbe7b-1983-4af1-ae5e-5fda58cbdbee">

## Get started
#### 1.the first step is clone this repo
    https://github.com/wjm666666/LMSSF.git
#### 2.Install all the required libraries for the experiment.
    pip install -r requirements.txt
#### 3.The process of data preprocessing 
    python3 data_process.py
#### 4.Training the model (without fusion model), parameter modifications at 291-295
    trainer.py
#### 5.Training a three-modal fusion model
    python3 trainer_multimodal.py
