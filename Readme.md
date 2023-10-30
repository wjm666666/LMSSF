####
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
####
    pip install -r requirements.txt
####
    python3 data_process.py
####
    python3 save_value.py
####
    python3 trainer_multimodal.py