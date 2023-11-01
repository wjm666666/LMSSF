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
