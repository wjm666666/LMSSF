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
<img width="880" alt="4424031a99009a43f6d2b60b8b9405b" src="https://github.com/wjm666666/LMSSF/assets/60913990/f3ecbe7b-1983-4af1-ae5e-5fda58cbdbee">

Use three feature clippers on the left to separate common and unique features between the two. The three common features are cropped again to distinguish trimodal common features and bimodal common features. Three-modal one-dimensional features are not processed. On the right side, the bimodal features
are fused with the unique features of another modality that do not intersect and are expressed in a two-dimensional
form, and the unique features of the three modalities are fused and expressed in a three-dimensional form. Below, the
three dimensional features are fed into the feature splicer for step-by-step calculation. In order to reduce the amount
of calculation, input is performed from low to high dimensions. When the predicted value reaches the set threshold
or the three-dimensional features are fully used, the results are output.
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
