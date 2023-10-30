from transformers import BertModel
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_


class BertTokenClassModel(nn.Module):

    def __init__(self, bert_path):
        super(BertTokenClassModel, self).__init__()
        self.model = BertModel.from_pretrained(bert_path)

        self.fc = nn.Sequential(
            nn.Linear(768, 128), nn.BatchNorm1d(128))

    def forward(self, text_array):
        x = self.model(input_ids=text_array["input_ids"],
                       attention_mask=text_array["attention_mask"])
        x = x.last_hidden_state
        x = x[:, -1, :]
        x = self.fc(x)
        return x


class MfccNet(nn.Module):
    def __init__(self, input_channel=32):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channel, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0))
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0))
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True))

        self.conv7 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True))

        self.conv8 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0))

        self.conv9 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True))

        self.conv10 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True))
        self.conv12 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0))
        self.conv13 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True))

        self.conv14 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True))

        self.conv15 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True))

        self.conv16 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0))

        self.fc17 = nn.Sequential(
            nn.Linear(int(512 * 12), 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5))

        self.fc18 = nn.Sequential(
            nn.Linear(4096, 128),
            nn.BatchNorm1d(128))

        self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7,
                          self.conv8, self.conv9, self.conv10, self.conv11, self.conv12, self.conv13, self.conv14,
                          self.conv15, self.conv16]

        self.fc_list = [self.fc17, self.fc18]
        self._init_weights()

    def forward(self, x):
        for conv in self.conv_list:
            x = conv(x)
        output = x.view(x.size()[0], -1)
        for fc in self.fc_list:
            output = fc(output)
        return output

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.normal_(0.0, 0.001)


class VGG(nn.Module):

    # initialize model
    def __init__(self, img_size=128, input_channel=1):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))

        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.conv13 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))

        self.conv14 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))

        self.conv15 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))

        self.conv16 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.fc17 = nn.Sequential(
            nn.Linear(int(512 * img_size * img_size / 32 / 32), 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5))

        self.fc18 = nn.Sequential(
            nn.Linear(4096, 128),
            nn.BatchNorm1d(128))

        self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7,
                          self.conv8, self.conv9, self.conv10, self.conv11, self.conv12, self.conv13, self.conv14,
                          self.conv15, self.conv16]

        self.fc_list = [self.fc17, self.fc18]
        self._init_weights()

    def forward(self, x):
        for conv in self.conv_list:
            x = conv(x)
        output = x.view(x.size()[0], -1)
        for fc in self.fc_list:
            output = fc(output)
        return output

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.normal_(0.0, 0.001)


class SubNet(nn.Module):
    def __init__(self, in_size, hidden_size, dropout):
        super(SubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))
        return y_3


class TextSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, bidirectional=False):
        super(TextSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           batch_first=True)

        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        _, final_states = self.rnn(x)
        h = final_states[0].squeeze()
        y_1 = self.linear_1(h)
        return y_1


class LMF(nn.Module):
    def __init__(self, input_dims, hidden_dims, text_out, dropouts, output_dim, rank):
        super(LMF, self).__init__()

        self.audio_in = input_dims[0]
        self.video_in = input_dims[1]
        self.text_in = input_dims[2]

        self.audio_hidden = hidden_dims[0]
        self.video_hidden = hidden_dims[1]
        self.text_hidden = hidden_dims[2]
        self.text_out = text_out
        self.output_dim = output_dim
        self.rank = rank

        self.audio_prob = dropouts[0]
        self.video_prob = dropouts[1]
        self.text_prob = dropouts[2]
        self.post_fusion_prob = dropouts[3]

        # define the pre-fusion subnetworks
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob)
        self.text_subnet = TextSubNet(self.text_in, self.text_hidden, self.text_out)

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.audio_factor = Parameter(torch.Tensor(self.rank, self.audio_hidden + 1, self.output_dim))
        self.video_factor = Parameter(torch.Tensor(self.rank, self.video_hidden + 1, self.output_dim))
        self.text_factor = Parameter(torch.Tensor(self.rank, self.text_out + 1, self.output_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))

        # init teh factors
        xavier_normal_(self.audio_factor)
        xavier_normal_(self.video_factor)
        xavier_normal_(self.text_factor)
        xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, audio_x, video_x, text_x):

        audio_h = self.audio_subnet(audio_x)
        video_h = self.video_subnet(video_x)
        text_h = self.text_subnet(text_x)
        batch_size = audio_h.data.shape[0]

        if audio_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _audio_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), audio_h), dim=1)
        _video_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), video_h), dim=1)
        _text_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), text_h), dim=1)

        fusion_audio = torch.matmul(_audio_h, self.audio_factor)
        fusion_video = torch.matmul(_video_h, self.video_factor)
        fusion_text = torch.matmul(_text_h, self.text_factor)
        fusion_zy = fusion_audio * fusion_video * fusion_text

        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        return output


# 第一层是输入的是三个模态的共同特征，
# 第二层输入的是每两个模态共同特征与第三个模态私有特征的向量积，
# 第三层输入是三个模态的私有特质的向量积，然后LMF降低的是第三层输入的那块的计算量


class FusionNet(nn.Module):
    def __init__(self, bert_path, c=128, classes_number=7):
        super(FusionNet, self).__init__()
        self.bert = BertTokenClassModel(bert_path=bert_path)
        self.mfcc = MfccNet()
        self.mfcc = MfccNet()
        self.vgg = VGG()
        self.lmf = LMF((128, 128, 16), (128, 128, 64), 32, (0.2, 0.2, 0.2, 0.5), output_dim=32, rank=4)

        self.common_text_first = nn.Sequential(nn.Linear(c, c), nn.BatchNorm1d(c))
        self.common_audio_first = nn.Sequential(nn.Linear(c, c), nn.BatchNorm1d(c))
        self.common_img_first = nn.Sequential(nn.Linear(c, c), nn.BatchNorm1d(c))

        self.private_text_first = nn.Sequential(nn.Linear(c, c), nn.BatchNorm1d(c))
        self.private_audio_first = nn.Sequential(nn.Linear(c, c), nn.BatchNorm1d(c))
        self.private_img_first = nn.Sequential(nn.Linear(c, c), nn.BatchNorm1d(c))

        self.first_out = nn.Linear(c * 3, classes_number)

        self.common_text_second = nn.Sequential(nn.Linear(c, c), nn.BatchNorm1d(c))
        self.common_audio_second = nn.Sequential(nn.Linear(c, c), nn.BatchNorm1d(c))
        self.common_img_second = nn.Sequential(nn.Linear(c, c), nn.BatchNorm1d(c))

        self.private_text_second = nn.Sequential(nn.Linear(c, c), nn.BatchNorm1d(c))
        self.private_audio_second = nn.Sequential(nn.Linear(c, c), nn.BatchNorm1d(c))
        self.private_img_second = nn.Sequential(nn.Linear(c, c), nn.BatchNorm1d(c))
        self.second_out = nn.Linear(c * 3, classes_number)
        self.third_out = nn.Linear(32, classes_number)

        self._init_weights()

    def forward(self, text_array, img_array, wav_array):
        bert = self.bert(text_array)
        wave = self.mfcc(wav_array)
        vgg = self.vgg(img_array)

        # first out
        bert_common_first = self.common_text_first(bert)
        wave_common_first = self.common_audio_first(wave)
        vgg_common_first = self.common_img_first(vgg)
        first_out = self.first_out(
            torch.cat(
                tensors=(bert_common_first, wave_common_first, vgg_common_first), dim=1))
        first_out = torch.softmax(first_out, dim=1)

        bert_private_first = self.private_text_first(bert)
        wave_private_first = self.private_audio_first(bert)
        vgg_private_first = self.private_img_first(bert)

        bert_common_second = self.common_text_second(bert_private_first)
        wave_common_second = self.common_audio_second(wave_private_first)
        vgg_common_second = self.common_img_second(vgg_private_first)

        bert_private_second = self.private_text_second(bert)
        wave_private_second = self.private_audio_second(bert)
        vgg_private_second = self.private_img_second(bert)

        common_text_wave = torch.tanh(bert_common_second + wave_common_second)
        common_text_img = torch.tanh(bert_common_second + vgg_common_second)
        common_wave_img = torch.tanh(wave_common_second + vgg_common_second)

        k1 = torch.matmul(common_text_wave.unsqueeze(1).permute(0, 2, 1), vgg_private_second.unsqueeze(1))
        k2 = torch.matmul(common_text_img.unsqueeze(1).permute(0, 2, 1), wave_private_second.unsqueeze(1))
        k3 = torch.matmul(common_wave_img.unsqueeze(1).permute(0, 2, 1), bert_private_second.unsqueeze(1))

        k1 = torch.mean(k1, dim=1)
        k2 = torch.mean(k2, dim=1)
        k3 = torch.mean(k3, dim=1)
        second_out = torch.cat(tensors=(k1, k2, k3), dim=1)

        second_out = self.second_out(second_out.reshape(-1, 384))
        second_out = torch.softmax(second_out, dim=1)

        third_out = self.lmf(vgg_private_second, wave_private_second, bert_common_second.reshape(-1, 8, 16))
        third_out = self.third_out(third_out)
        third_out = torch.softmax(third_out, dim=1)

        return first_out, second_out, third_out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.normal_(0.0, 0.001)


if __name__ == '__main__':
    bert_path_ = r'../data/bert-base-text-english'
    wa_array = torch.randn(2, 32, 400)
    text = {"input_ids": torch.tensor([[345, 232, 13, 544, 2323], [345, 232, 13, 544, 2323]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])}
    im_array = torch.randn(2, 1, 128, 128)
    model = FusionNet(bert_path_)
    model.eval()
    y = model(text, im_array, wa_array)
    print(y[0].shape)
    print(y[1].shape)
    print(y[2].shape)

