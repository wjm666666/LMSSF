import librosa
import torch
from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2Processor
import os
import numpy as np
from tqdm import tqdm
import pickle


class ProcessWaveImgTextSavePkl(object):
    def __init__(self, expression_file, wave_file, save_file):

        self.expression_file = expression_file
        self.wave_file = wave_file
        self.save_file = save_file

        self.wav_index_to_target = {"01": "neutral", "02": "contempt", "03": "happy", "04": "sadness",
                                    "05": "anger", "06": "fear", "07": "disgust", "08": "surprise"}

    def process_expression(self):
        print("\nStart processing expression data...")

        count = {"anger": 0, "contempt": 0, "disgust": 0,
                 "fear": 0, "happy": 0, "sadness": 0, "surprise": 0}
        file_list = []
        expression_list = os.listdir(self.expression_file)

        for i, target in zip(
                tqdm(
                    range(
                        len(expression_list))
                ),
                expression_list):

            target_file = os.path.join(self.expression_file, target)
            img = os.listdir(target_file)
            for expression_img in img:
                expression_file = os.path.join(target, expression_img)
                file_list.append(expression_file)
                count[str(target)] += 1

        np.random.seed(1)
        np.random.shuffle(file_list)
        return file_list, count

    def process_audio_and_text(self):

        expression_list, expression_count = self.process_expression()
        print("Start processing audio data...")

        wav_container = {"neutral": [], "anger": [], "contempt": [], "disgust": [],
                         "fear": [], "happy": [], "sadness": [], "surprise": []}

        actor_list = os.listdir(self.wave_file)

        for i, actor in zip(
                tqdm(
                    range(
                        len(actor_list))), actor_list):

            wav_file = os.path.join(self.wave_file, actor)

            wave = os.listdir(wav_file)
            for audio_wav in wave:
                label_num = audio_wav.split("-")[2]
                label_str = self.wav_index_to_target[label_num]

                wav_abspath = os.path.join(wav_file, audio_wav)
                text = self.retrained_audio_to_text(wav_file=wav_abspath)
                actor_abspath = os.path.join(actor, audio_wav)

                information = {"actor_file": str(actor_abspath), "text": text, "label": label_str}
                wav_container[label_str].append(information)

        del wav_container["neutral"]
        for key in wav_container.keys():
            expression_value_len = expression_count[key]
            container_value = self.shuffle(wav_container[key])
            container_value = container_value * 10
            container_value = container_value[:expression_value_len]
            wav_container[key] = container_value

        pkl = {"expression": expression_list, "wav_text": wav_container}
        self.save_pickle(pkl=pkl, pickle_file=self.save_file)

    @staticmethod
    def retrained_audio_to_text(wav_file):
        processor = Wav2Vec2Processor.from_pretrained(
            pretrained_model_name_or_path="jonatasgrosman/wav2vec2-large-xlsr-53-english")
        model = Wav2Vec2ForCTC.from_pretrained(
            pretrained_model_name_or_path="jonatasgrosman/wav2vec2-large-xlsr-53-english")
        audio_input, sample_rate = librosa.load(wav_file, sr=16000)
        input_values = processor(audio_input,
                                 sampling_rate=sample_rate,
                                 return_tensors="pt").input_values
        logit = model(input_values).logits
        predicted_ids = torch.argmax(logit, dim=-1)
        transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
        return transcription

    @staticmethod
    def shuffle(data):
        np.random.seed(1)
        np.random.shuffle(data)
        return data

    @staticmethod
    def save_pickle(pkl, pickle_file):
        file = open(pickle_file, 'wb')
        pickle.dump(pkl, file)
        file.close()


if __name__ == '__main__':
    tool = ProcessWaveImgTextSavePkl(expression_file='dataset/CK+',
                                     wave_file='dataset/RAVDESS',
                                     save_file='processed_pkl/feature.pkl')
    tool.process_audio_and_text()
