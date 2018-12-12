import os, time
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import numpy as np
import librosa as lr
import bisect
import math
from pylab import*
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from wavenet_vocoder.util import inv_mulaw_quantize
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(1)
#seq_len = 20
hidden_dim = 896
input_dim  = 128
train_epoch= 60
batch_size = 15
seq_len    = 32
target_len = 1
out_classes= 256
lr         = 0.0002
threshold  = 0.5 
nouse_lastmodel = False
model_epoch = 600
model_step  = 401
# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generator, self).__init__()
        self.deconv1 = nn.ConvTranspose1d(80*2, d*8, 4, 2, 1)
        self.deconv1_bn = nn.BatchNorm1d(d*8)
#        self.deconv1_2 = nn.ConvTranspose2d(10, d*4, 4, 2, 1)
#        self.deconv1_2_bn = nn.BatchNorm2d(d*2)
        self.deconv2 = nn.ConvTranspose1d(d*8, d*6, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm1d(d*6)
        self.deconv3 = nn.ConvTranspose1d(d*6, d*6, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm1d(d*6)
        self.deconv4 = nn.ConvTranspose1d(d*6, d*4, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm1d(d*4)
        self.deconv5 = nn.ConvTranspose1d(d*4, d*4, 4, 2, 1)
        self.deconv5_bn = nn.BatchNorm1d(d*4)
        self.deconv6 = nn.ConvTranspose1d(d*4, d*2, 4, 2, 1)
        self.deconv6_bn = nn.BatchNorm1d(d*2)
        self.deconv7 = nn.ConvTranspose1d(d*2, d*2, 4, 2, 1)
        self.deconv7_bn = nn.BatchNorm1d(d*2)
        self.deconv8 = nn.ConvTranspose1d(d*2, 1, 4, 2, 1)
        self.deconv8_bn = nn.BatchNorm1d(d*1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        input = input.transpose(1,2)
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
#        y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)))
#        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.relu(self.deconv5_bn(self.deconv5(x)))
        x = F.relu(self.deconv6_bn(self.deconv6(x)))
        x = F.relu(self.deconv7_bn(self.deconv7(x)))
#        x = F.relu(self.deconv8_bn(self.deconv8(x)))/

        x = torch.tanh(self.deconv8(x))
        # x = F.relu(self.deconv4_bn(self.deconv4(x)))
        # x = F.tanh(self.deconv5(x))

        return x

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(discriminator, self).__init__()
        
        self.unconv1    = nn.ConvTranspose1d(80,256,16,4,6)
        self.unconv_act1= nn.Tanh()
        self.unconv2    = nn.ConvTranspose1d(256,256,16,4,6)
        self.unconv_act2= nn.Tanh()
        
        
        self.conv1    = nn.Conv1d(1, d, 4, 2,1)
        self.conv1_bn = nn.BatchNorm1d(d)
        self.conv2    = nn.Conv1d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm1d(d*2)
        self.conv3    = nn.Conv1d(d*2, d*2, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm1d(d*2)
        self.conv4    = nn.Conv1d(d*2, d*4, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm1d(d*4)
        self.conv5    = nn.Conv1d(d*4+256, d*4+256, 4, 2, 1)
        self.conv5_bn = nn.BatchNorm1d(d*4+256)
        self.conv6    = nn.Conv1d(d*4+256, d*6, 4, 2, 1)
        self.conv6_bn = nn.BatchNorm1d(d*6)
        self.conv7    = nn.Conv1d(d*6, d*4, 4, 2, 1)
        self.conv7_bn = nn.BatchNorm1d(d*4)
        self.conv8    = nn.Conv1d(d*4, d*4, 8, 4, 3)
        self.conv8_bn = nn.BatchNorm1d(d*4)
        self.conv9    = nn.Conv1d(d*4, d*2, 8, 4, 3)
        self.conv9_bn = nn.BatchNorm1d(d*2)
        self.conv10    = nn.Conv1d(d*2, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, wav, mel):
        mel =mel.transpose(1,2)
        mel = self.unconv_act1(self.unconv1(mel))
        mel = self.unconv_act2(self.unconv2(mel))
        
        
        x = F.leaky_relu(self.conv1_bn(self.conv1(wav)), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = torch.cat([x,mel],1)
        x = F.leaky_relu(self.conv5_bn(self.conv5(x)), 0.2)
        x = F.leaky_relu(self.conv6_bn(self.conv6(x)), 0.2)
        x = F.leaky_relu(self.conv7_bn(self.conv7(x)), 0.2)
        x = F.leaky_relu(self.conv8_bn(self.conv8(x)), 0.2)
        x = F.leaky_relu(self.conv9_bn(self.conv9(x)), 0.2)
        x = torch.sigmoid(self.conv10(x))
        
        return x

class DataSet(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_file,
                 wave_file,
                 item_length,
                 target_length,
                 file_location=None,
                 classes=256,
                 sampling_rate=16000,
                 mono=True,
                 normalize=False,
                 dtype=np.uint8,
                 train=True,
                 test_stride=100):


        self.dataset_file = dataset_file
        self._item_length = item_length
        self._test_stride = test_stride
        self.target_length = target_length
        self.classes = classes
        self.wave_file = wave_file

        if not os.path.isfile(dataset_file):
            assert file_location is not None, "no location for dataset files specified"
            self.mono = mono
            self.normalize = normalize

            self.sampling_rate = sampling_rate
            self.dtype = dtype
        else:
            # Unknown parameters of the stored dataset
            # TODO Can these parameters be stored, too?
            self.mono = None
            self.normalize = None

            self.sampling_rate = None
            self.dtype = None

        self.data = np.load(self.dataset_file, mmap_mode='r')
        self.start_samples = [0]
        self._length = 0
        self.calculate_length()
        self.train = train


    def calculate_length(self):
        start_samples = [0]
        for i in range(len(self.data.keys())):
            start_samples.append(start_samples[-1] + len(self.data['arr_' + str(i)]))
        available_length = start_samples[-1] - (self._item_length - (self.target_length - 1)) - 1   #start_samples[-1] 6902617
        self._length = math.floor(available_length / self.target_length)
        self.start_samples = start_samples

    def set_item_length(self, l):
        self._item_length = l
        self.calculate_length()

    def __getitem__(self, idx):
#        idx = 301675
        if self._test_stride < 2:
            sample_index = idx * self.target_length
        elif self.train:
            sample_index = idx * self.target_length + math.floor(idx / (self._test_stride-1))
        else:
            sample_index = self._test_stride * (idx+1) - 1

        file_index = bisect.bisect_left(self.start_samples, sample_index) - 1
        if file_index < 0:
            file_index = 0
        if file_index + 1 >= len(self.start_samples):
            print("error: sample index " + str(sample_index) + " is to high. Results in file_index " + str(file_index))
        position_in_file = sample_index - self.start_samples[file_index]
        
        end_position_in_next_file = sample_index + self._item_length - 1 - self.start_samples[file_index + 1]
        
#        position_in_wavefile = position_in_file*275

        if end_position_in_next_file < 0:
            file_name = 'arr_' + str(file_index)
            this_file = np.load(self.dataset_file, mmap_mode='r')[file_name]
            if (position_in_file + self._item_length) == len(this_file):
                position_in_file = position_in_file-1
            sample = this_file[position_in_file:position_in_file + self._item_length]
            
            position_in_wavefile = position_in_file*256
            wave_file = np.load(self.wave_file, mmap_mode='r')[file_name]
            wave_sample = wave_file[position_in_wavefile:position_in_wavefile + self._item_length*256]
        else:
            # load from two files
            position_in_file = position_in_file - self._item_length -2
            position_in_wavefile = position_in_file*256
                        
            file_name = 'arr_' + str(file_index)
            this_file = np.load(self.dataset_file, mmap_mode='r')[file_name]
            sample = this_file[position_in_file:position_in_file + self._item_length]
            
            wave_file = np.load(self.wave_file, mmap_mode='r')[file_name]
            wave_sample = wave_file[position_in_wavefile:position_in_wavefile + self._item_length*256]

        mel_example      = torch.from_numpy(sample).type(torch.FloatTensor)
        wave_example = torch.from_numpy(wave_sample).type(torch.FloatTensor)
        wave_example = wave_example/128.-1.
        return wave_example, mel_example

    def __len__(self):
        test_length = math.floor(self._length / self._test_stride)
        if self.train:
            return self._length - test_length
        else:
            return test_length

def save_image(wav,epoch,batch,image_len = 200,save_path = 'waveGAN_results/'):
    timeArray = np.arange(0, len(wav), 1)   #[0s, 1s], 5060个点
    timeArray = timeArray / 22050   #[0s, 0.114s]
    timeArray = timeArray * 1000       #[0ms, 114ms]
#    plt.switch_backend('agg')
    plt.figure(figsize=(image_len, 5))
    plt.plot(timeArray, wav, color='b')
    ylabel('Amplitude')
    xlabel('Time (ms)')
    plt.savefig(save_path+f"test_epoch_{epoch}_batch_{batch}.png")

img_size = 32
transform = transforms.Compose([
        transforms.Scale(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
data = DataSet(dataset_file = 'data/mel_datas.npz',
                      wave_file    = 'data/wave_datas.npz',
                      item_length  = seq_len,
                      target_length= target_len,
                      test_stride  = 500)

print('the dataset has ' + str(len(data)) + ' items',str(len(data)/batch_size) + ' batch_step')  
print('train_GAN','time:',time.asctime(time.localtime(time.time())).split()[2:4],'start training...')

train_loader = torch.utils.data.DataLoader(data,
    batch_size=batch_size, shuffle=True,num_workers=16)

# network
if nouse_lastmodel:
    G = generator(128)
    D = discriminator(128)
    G.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)
else:
    D = torch.load(f'model_gan/wD_epoch{model_epoch}step{model_step}.model', map_location='cpu')
    G = torch.load(f'model_gan/wG_epoch{model_epoch}step{model_step}.model', map_location='cpu')
    print(f'load last model epoch{model_epoch}step{model_step}')
if use_cuda:
    G.cuda()
    D.cuda()
else:
    G.cpu()
    D.cpu()


mel = np.load('/Users/edz/Desktop/work/tjTaco2/tacotron_output/gta/speech-mel-{0:0>5}.npy'.format(4))
mel = mel/6
mel = torch.FloatTensor(mel).unsqueeze(0)
z_ = torch.randn((1, mel.shape[1],80))
G_input    = torch.cat([z_,mel],2)
G_result = G(G_input)
G_result2=G_result
G_result = (G_result.view(-1)+1)*128

G_result = G_result.type(torch.LongTensor)
mu_gen = inv_mulaw_quantize(G_result.detach().numpy(), 256)
wav_name = f'waveGAN_results/{model_epoch}epo.wav'
import librosa

mu_gen = G_result.type(torch.FloatTensor).detach().numpy()/128.-1.

librosa.output.write_wav(wav_name, mu_gen, sr=22050)
save_image(mu_gen[:20000],model_epoch,2,image_len = 400)

