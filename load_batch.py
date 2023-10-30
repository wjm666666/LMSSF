import os.path
from PIL import Image
import numpy as np
from torchvision import transforms as transforms
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional
import pickle
import librosa
from transformers import BertTokenizer


def split_train_test(pkl_path, k=0.7):
    def load_pkl(pkl_file):
        file = open(pkl_file, 'rb')
        pkl_data = pickle.load(file)
        return pkl_data

    def count(counted, state: str):
        from collections import Counter
        dataset = []
        for v in counted.keys():
            dataset += counted[v]
        dataset = [x['label'] for x in dataset]
        print(f"{state} Set: {Counter(dataset)}")

    def count_ex(counted):
        from collections import Counter
        dataset = [x.split("\\")[0] for x in counted]
        return dict(Counter(dataset))

    def correspond(data_set):
        targets = {'anger': [], 'contempt': [], 'disgust': [], 'fear': [],
                   'happy': [], 'sadness': [], 'surprise': []}
        expression_list = data_set["expression"]
        wave_txt = data_set["wave_text"]
        for name in expression_list:
            names = name.split("\\")[0]
            targets[names].append(name)

        expressions = []
        audio_text = []
        for label in targets.keys():
            expressions += targets[label]
            audio_text += wave_txt[label]

        return {"expression": expressions, "wave_text": audio_text}

    train_test_data = load_pkl(pkl_file=pkl_path)
    expression = train_test_data["expression"]
    train_expression = expression[:int(len(expression) * k)]
    test_expression = expression[int(len(expression) * k):]

    train_count = count_ex(counted=train_expression)

    train_wav, test_wav = {}, {}
    wave_text = train_test_data['wav_text']

    for key in wave_text.keys():
        train_wav[key] = wave_text[key][:train_count[key]]
        test_wav[key] = wave_text[key][train_count[key]:]

    count(counted=train_wav, state='Train')
    count(counted=test_wav, state='Test')

    train = {"expression": train_expression, "wave_text": train_wav}
    test = {"expression": test_expression, "wave_text": test_wav}

    train = correspond(train)
    test = correspond(test)
    return train, test


class BatchDataLoadImgWavText(Dataset):
    def __init__(self,
                 batch_data: Optional[dict],
                 expression_file: Optional[str],
                 wave_file: Optional[str],
                 mfcc_norm_way: Optional[str],
                 state='train',
                 mfcc_num=32,
                 mfcc_length=400,
                 json_value_file='data/processed_pkl/value.json'
                 ):
        super(BatchDataLoadImgWavText, self).__init__()

        assert mfcc_norm_way in ['max', 'max-min', 'mean-std', None]

        self.batch_data = batch_data
        self.wave_file = wave_file
        self.mfcc_norm_way = mfcc_norm_way
        self.state = state
        self.mfcc_num = mfcc_num
        self.mfcc_length = mfcc_length
        self.expression_file = expression_file
        self.tokenizer = BertTokenizer.from_pretrained("data/bert-base-text-english")

        json_value = eval(self.load_json_value(json_file=json_value_file))

        self.mfcc_max_value = json_value["max_value"]
        self.mfcc_min_value = json_value["min_value"]
        self.mfcc_mea_value = json_value["mean_value"]
        self.mfcc_std_value = json_value["std_value"]

        self.expression = self.batch_data["expression"]
        self.wave_text = self.batch_data["wave_text"]

        if self.state == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.RandomCrop(size=128),
                transforms.RandomHorizontalFlip(p=0.2),
                transforms.RandomVerticalFlip(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])])

        if self.state == 'test':
            self.transforms = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])])

        self.unite_label_to_index = {"anger": 0, "contempt": 1, "disgust": 2, "fear": 3,
                                     "happy": 4, "sadness": 5, "surprise": 6}

        self.empty = np.zeros(shape=(self.mfcc_num, self.mfcc_length))

    def __getitem__(self, index):

        expression = self.expression[index]
        audio_dict = self.wave_text[index]

        expression_file = os.path.join(self.expression_file, expression)
        expression_img = Image.open(expression_file)
        img_array = expression_img.convert('L')
        img_array = self.transforms(img_array)

        audio_file = os.path.join(self.wave_file, audio_dict["actor_file"])
        mfcc_array = self.mfcc(wav_file=audio_file, n_mfcc=self.mfcc_num)
        mfcc_array = np.hstack(tup=(mfcc_array, self.empty))
        mfcc_array = mfcc_array[:, :self.mfcc_length]

        if self.mfcc_norm_way == 'max':
            mfcc_array = mfcc_array / self.mfcc_max_value
        elif self.mfcc_norm_way == 'max-min':
            mfcc_array = (mfcc_array - self.mfcc_min_value) / (self.mfcc_max_value - self.mfcc_min_value)
        elif self.mfcc_norm_way == 'mean-std':
            mfcc_array = (mfcc_array - self.mfcc_mea_value) / self.mfcc_std_value
        else:
            assert self.mfcc_norm_way is None
            mfcc_array = mfcc_array
        mfcc_array = torch.tensor(mfcc_array, dtype=torch.float32)

        text_str = audio_dict["text"]
        text_array = self.text_encode(tokenizer=self.tokenizer, text=text_str)

        label = self.unite_label_to_index[audio_dict["label"]]
        label = torch.tensor(int(label))

        return img_array, mfcc_array, text_array["input_ids"], text_array["attention_mask"], label

    def __len__(self):
        return len(self.expression)

    @staticmethod
    def mfcc(wav_file, n_mfcc):
        y, sr = librosa.load(wav_file, sr=None)
        return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    @staticmethod
    def load_json_value(json_file):
        with open(json_file, 'r', encoding='utf8') as f:
            value = f.read()
        return value

    @staticmethod
    def text_encode(tokenizer, text):
        words = tokenizer.encode_plus(text=text,
                                      max_length=16,
                                      padding='max_length',
                                      return_tensors='pt')
        return {"input_ids": words['input_ids'].squeeze(0),
                "attention_mask": words['attention_mask'].squeeze(0)}


if __name__ == '__main__':

    train_set, test_set = split_train_test(pkl_path='data/processed_pkl/feature.pkl')

    data = BatchDataLoadImgWavText(batch_data=train_set,
                                   expression_file='data/dataset/CK+',
                                   wave_file='data/dataset/RAVDESS',
                                   state='train',
                                   mfcc_norm_way="max")
    dataloader = DataLoader(data, batch_size=20, shuffle=True)

    for i, batch in enumerate(dataloader):
        img_feature, mfcc_feature, input_ids, attention_mask, target = batch
        print('\nimg feature shape:', img_feature.shape)
        print('mfcc array shape:', mfcc_feature.shape)
        print('bert input ids:', input_ids.shape)
        print('bert attention mask:', attention_mask.shape)
        print('target array:', target)
        if i == 2: break
