import json
import numpy as np
import librosa
from tqdm import tqdm
import os


def save_json(value_json_file, data):
    with open(value_json_file, 'w', encoding='utf8') as f:
        f.write(
            json.dumps(
                data,
                indent=4,
                separators=(', ', ': '),
                ensure_ascii=False))


def mfcc(wav_file, n_mfcc):
    y, sr = librosa.load(wav_file, sr=None)
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)


def read_wave(data_file, n_mfcc, json_value_file, length=400):
    features = []
    actor_list = os.listdir(data_file)

    for i, actor in zip(
            tqdm(
                range(
                    len(actor_list))), actor_list):

        wav_file = os.path.join(data_file, actor)

        wave = os.listdir(wav_file)
        empty = np.zeros(shape=(n_mfcc, length))
        for audio_wav in wave:
            wav_abspath = os.path.join(wav_file, audio_wav)
            feature = mfcc(wav_file=wav_abspath, n_mfcc=n_mfcc)
            feature = np.hstack(tup=(feature, empty))
            feature = feature[:, :length]
            features.append(feature)
    features = np.array(features, dtype=object)

    # max  280.1737060546875
    # min  -1085.479736328125
    # mean -15.889966703914636
    # std  106.11911783836439

    value = {"max_value": np.max(features), "min_value": np.min(features),
             "mean_value": np.mean(features), "std_value": np.std(features)}

    save_json(value_json_file=json_value_file, data=value)


if __name__ == '__main__':
    read_wave('dataset/RAVDESS',
              json_value_file='processed_pkl/value.json',
              n_mfcc=32)
