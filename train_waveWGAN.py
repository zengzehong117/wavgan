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
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(0)
#seq_len = 20
hidden_dim = 896
input_dim  = 128
train_epoch= 501
batch_size = 51
seq_len    = 32
target_len = 1
out_classes= 256
lr         = 0.0002
threshold  = 0.5 
nouse_lastmodel = False
model_epoch = 300
model_step  = 91
LAMBDA = 0.3
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

        x = self.deconv8(x)
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
        self.conv5    = nn.Conv1d(d*4+256, d*8, 4, 2, 1)
        self.conv5_bn = nn.BatchNorm1d(d*8)
        self.conv6    = nn.Conv1d(d*8, d*6, 4, 2, 1)
        self.conv6_bn = nn.BatchNorm1d(d*6)
        self.conv7    = nn.Conv1d(d*6, d*4, 4, 2, 1)
        self.conv7_bn = nn.BatchNorm1d(d*4)
        self.conv8    = nn.Conv1d(d*4, d*2, 4, 2, 1)
        self.conv8_bn = nn.BatchNorm1d(d*2)
        self.conv9    = nn.Conv1d(d*2, 1, 3, 1, 1)
        

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, wav, mel):
        mel =mel.transpose(1,2)
        mel = self.unconv_act1(self.unconv1(mel))
        mel = self.unconv_act2(self.unconv2(mel))
        
        
        x = F.leaky_relu(self.conv1(wav), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = torch.cat([x,mel],1)
        x = F.leaky_relu(self.conv5(x), 0.2)
        x = F.leaky_relu(self.conv6(x), 0.2)
        x = F.leaky_relu(self.conv7(x), 0.2)
        x = F.leaky_relu(self.conv8(x), 0.2)
#        x = F.leaky_relu(self.conv9(x), 0.2)
        x = self.conv9(x)
        
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

        self.data = np.load(self.dataset_file, mmap_mode='r')
        self.start_samples = [0]
        self._length = 0
        self.train = train

    def __getitem__(self, idx):

        file_index = idx
        file_name = 'arr_' + str(file_index)
        mel_file = np.load(self.dataset_file, mmap_mode='r')[file_name]
        wave_file = np.load(self.wave_file, mmap_mode='r')[file_name]
        mel_example      = torch.from_numpy(mel_file).type(torch.FloatTensor)
        wave_example     = torch.from_numpy(wave_file).type(torch.FloatTensor)
        wave_example     = wave_example/128.-1.

        return wave_example,mel_example

    def __len__(self):
        test_length = len(self.data.keys())
        return test_length


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


#audio_sequence=wav
#sd.play(wav/10,22050)



def save_image(wav,epoch,batch,image_len = 200,save_path = 'test_gan/'):
    timeArray = np.arange(0, len(wav), 1)   #[0s, 1s], 5060个点
    timeArray = timeArray / 22050   #[0s, 0.114s]
    timeArray = timeArray * 1000       #[0ms, 114ms]
    plt.switch_backend('agg')
    plt.figure(figsize=(image_len, 3))
    plt.plot(timeArray, wav, color='b')
    ylabel('Amplitude')
    xlabel('Time (ms)')
    plt.savefig(save_path+f"test_epoch_{epoch}_batch_{batch}.png")


def calc_gradient_penalty(netD, real_data, fake_data,mel):
    #print real_data.size()
    alpha = torch.rand(batch_size//3, 1,1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates,mel)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

img_size = 32
transform = transforms.Compose([
        transforms.Scale(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
data = DataSet(dataset_file = 'data/mels_pad.npz',
                      wave_file    = 'data/wavs_pad.npz',
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


# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
one = torch.tensor(1.)
mone = one * -1
if use_cuda:
    one = one.cuda()
    mone = mone.cuda()
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

onehot = torch.zeros(10, 10)
onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1)
fill = torch.zeros([10, 10, img_size, img_size])
for i in range(10):
    fill[i, i, :, :] = 1

print('training start!')
start_time = time.time()
start_epoch = 0 if nouse_lastmodel else model_epoch+1
for epoch in range(start_epoch,train_epoch):
    D_losses = []
    G_losses = []

    # learning rate decay
    if (epoch+1) == 40:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    if (epoch+1) == 50:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    epoch_start_time = time.time()
    y_real_ = torch.ones(batch_size//3)
    y_fake_ = torch.zeros(batch_size//3)
    if use_cuda:
        y_real_, y_fake_ = Variable(y_real_.cuda()), Variable(y_fake_.cuda())
    else:
        y_real_, y_fake_ = Variable(y_real_.cpu()), Variable(y_fake_.cpu())
    batch = 0
    tmp_loss1 = 0.
    tmp_loss2 = 0.
    tmp_loss3 = 0.
    publish_l = 0.
    d_fake_l  = 0.
    d_real_l  = 0.
    real_lr   = 1.
    for wav,mel  in train_loader:
        if batch%5 == 0:
            for p in D.parameters():  # reset requires_grad
                p.requires_grad = True 
        if mel.size()[0] != batch_size:
            continue
        if use_cuda:
            wav = Variable(wav.cuda())
            mel = Variable(mel.cuda())/6
        else:
            wav = Variable(wav.cpu())
            mel = Variable(mel.cpu())/6
        D_mel = mel[:batch_size//3]
        D_wav = wav[:batch_size//3]
        G_mel1= mel[batch_size//3:batch_size//3*2]
        G_mel2= mel[batch_size//3*2:batch_size]        # train discriminator D
        D.zero_grad()
        mini_batch = D_mel.size()[0]
        seq_mel_len= D_mel.size()[1]

        if batch%1 == 0:
            D.zero_grad()
            mini_batch = D_mel.size()[0]
            seq_mel_len= D_mel.size()[1]
            D_wav = D_wav.unsqueeze(1)
            D_real_result = D(D_wav, D_mel).squeeze()
            D_real_result = D_real_result.mean()
            D_real_result.backward(mone*real_lr)
            z_ = torch.randn((mini_batch, seq_mel_len,80)).view(-1, seq_mel_len, 80)
            if use_cuda:
                z_ = Variable(z_.cuda())
            else:
                z_ = Variable(z_.cpu())
            G_input    = torch.cat([z_,D_mel],2)

            G_result = G(G_input)
            G_result1=G_result
#        D_result = D(G_result, y_fill_).squeeze()
            D_fake_result = D(G_result, D_mel)
            D_fake_result = D_fake_result.mean()
            D_fake_result.backward(one)
            if D_real_result.item()<100.:
                real_lr = 1.002
            elif D_fake_result.item()>-100.:
                real_lr = 0.998
            else:
                real_lr = 1.
        
            gradient_penalty = calc_gradient_penalty(D, D_wav.data, G_result1.data,D_mel)
            gradient_penalty.backward()
            D_cost = D_fake_result - D_real_result + gradient_penalty
            Wasserstein_D = D_real_result - D_fake_result
            D_optimizer.step()
            #print(D_cost.item())

        if batch%1 == 0:
            for p in D.parameters():
                p.requires_grad = False          
            G.zero_grad()

            z_ = torch.randn((mini_batch, seq_mel_len,80)).view(-1, seq_mel_len, 80)
        
            if use_cuda:
                z_ = Variable(z_.cuda())
            else:
                z_ = Variable(z_.cpu())
            G_input    = torch.cat([z_,G_mel2],2)

            G_result = G(G_input)
            D_result = D(G_result, G_mel2).squeeze()
            D_result = D_result.mean()
            D_result.backward(mone)
            G_cost = -D_result
            #print(G_cost.item())
            G_optimizer.step()
            for p in D.parameters():  # reset requires_grad
                p.requires_grad = True

            tmp_loss1  += D_cost.item()
            tmp_loss2  += G_cost.item()
            tmp_loss3  += Wasserstein_D.item()
            publish_l  += gradient_penalty.item()
            d_fake_l   += D_fake_result.item()
            d_real_l   += D_real_result.item()

        batch += 1
        out_num = 90
        if batch%out_num == 0:
            print('time:',time.asctime( time.localtime(time.time())).split()[3],'epoch:',epoch,'batch:',batch,'D_l:',round(tmp_loss1/out_num,3),'G_l:',round(tmp_loss2/out_num,3),'WD:',round(tmp_loss3/out_num,3),'p_m:',round(publish_l/out_num,3),'D_fakel:',round(d_fake_l/out_num,3),'D_realresult:',round(d_real_l/out_num,3))
            if batch%(out_num) == 0:
                save_image(G_result[0][0].cpu().detach().numpy()[:25000],epoch,batch+1,image_len = 400)
                save_image(wav[batch_size//3*2].cpu().detach().numpy()[:25000],epoch,batch+2,image_len = 400)
            if batch%(out_num) == 0 and epoch%20==0:
                torch.save(G,'model_gan/wG_epoch%dstep%d.model'%(epoch,batch+1))
                torch.save(D,'model_gan/wD_epoch%dstep%d.model'%(epoch,batch+1))
            
            tmp_loss1 = 0.
            tmp_loss2 = 0.
            tmp_loss3 = 0.
            publish_l = 0.
            d_fake_l  = 0.
            d_real_l  = 0.
    

