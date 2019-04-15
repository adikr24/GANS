import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms

os.chdir('C:\\Users\\adity\\Desktop\\udacity_deep\\data')

def transform_image(size):
    transform= transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()])
    return transform

#size= 32
def return_loader(size,batch_size):
    train_data= datasets.SVHN('/SVHN_data',transform=transform_image(size),download=True)
    train_loader= torch.utils.data.DataLoader(train_data,batch_size)
    return train_loader


size=32
batch_size=10
data_loader=return_loader(size,batch_size)

img,labels=next(iter(data_loader))

img.size()
labels.size()


##plot image
def implot(image):
    im1=np.transpose(image,(1,2,0))
    return plt.imshow(im1)

im1=img[0]
implot(im1)    


def conv(in_channels,out_channels,kernel_size=4,stride=2,padding=1,batch_norm=True):
    layers=[]
    conv_layers= nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                           kernel_size=kernel_size,stride=stride,padding=padding,bias=False)
    layers.append(conv_layers)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

class Discriminator(nn.Module):
    def __init__(self,conv_dim):
        super(Discriminator,self).__init__()
        self.conv_dim=conv_dim
        self.conv1 =conv(3,conv_dim,4,batch_norm=False)
        self.conv2= conv(conv_dim,conv_dim*2,4)
        self.conv3=conv(conv_dim*2,conv_dim*4,4)
        self.fc1= nn.Linear(conv_dim*4*4*4,1)
    
    def forward(self,x):
        x=F.leaky_relu(self.conv1(x),0.2)
        x=F.leaky_relu(self.conv2(x),0.2)
        x=F.leaky_relu(self.conv3(x),0,2)
        x= x.view(-1,self.conv_dim*4*4*4)
        x= self.fc1(x)
        return x


D=Discriminator(28)
out=D.forward(img)
conv(3,64,4)


def deconv(in_channels,out_channels,kernel_size,stride=2,padding=1,batch_norm=True):
    layers= []
    transpose_conv_layer= nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride,
                                             padding,bias=False)
    layers.append(transpose_conv_layer)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)



class Generator(nn.Module):
    def __init__(self, z_size, conv_dim=32):
        super(Generator, self).__init__()
        self.conv_dim = conv_dim
        self.fc = nn.Linear(z_size, conv_dim*4*4*4)
        self.t_conv1 = deconv(conv_dim*4, conv_dim*2, 4)
        self.t_conv2 = deconv(conv_dim*2, conv_dim, 4)
        self.t_conv3 = deconv(conv_dim, 3, 4, batch_norm=False)
    def forward(self, x):
        out = self.fc(x)
        out = out.view(-1, self.conv_dim*4, 4, 4) # (batch_size, depth, 4, 4)
        out = F.relu(self.t_conv1(out))
        out = F.relu(self.t_conv2(out))
        out = self.t_conv3(out)
        out = F.tanh(out)
        
        return out

############################
## defining the loses
def real_loss(D_out,smooth=False):
    batch_size= D_out.size(0)
    if smooth:
        labels.torch.ones(batch_size)*0.9
    else:
        labels= torch.ones(batch_size)
    
    if train_on_gpu:
        labels= labels.cuda()
        
    criterion= nn.BCEWithLogitsLoss()
    loss= criterion(D_out.squeeze(),labels)
    return loss
    
    
import torch.cuda
train_on_gpu=torch.cuda.is_available()    



def fake_loss(D_out):
    batch_size= D_out.size(0)
    labels= torch.zeros(batch_size)
    if train_on_gpu:
        labels= labels.cuda()
    criterion= nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(),labels)
    return loss


####################################

def rev_scale(images,resize=(1,-1)):
    #image= image*(resize.max()-resize.min()) - resize.max()
    image= images.detach().numpy()
    image= ((image+ max(resize))/ (max(resize)-min(resize)))
    return image

def plot_generator(image,pic_number):
    #im= image.detach().numpy()
    if type(image)==torch.Tensor:
        image= image.detach().numpy()
    im=np.transpose(image[pic_number],(1,2,0))
    return plt.imshow(im)


##### parameters
conv_dim=64
batch_size=32

#################################generate a distribution for Generator
z_size=100
z= np.random.uniform(-1,1,size=(batch_size,z_size))
z= torch.from_numpy(z).float()

### instantiate class
D= Discriminator(conv_dim)
G=Generator(z_size= z_size,conv_dim=conv_dim)


####################################

images,labels=next(iter(data_loader))
D_out= D.forward(images)
G_out=G.forward(z)

## original_image
plot_generator(images,1)
## generated images
## since tan function is used we'll reverse it
gg= rev_scale(G_out,resize=(1,-1))
plot_generator(gg,1)
#########################
## training the generator and discriminator on the images received
##without the loop just for the 32 images we'' train the generator

#### passing the learning Params
import torch.optim as optim
lr= 0.005
beta1= 0.5
beta2= 0.9999

d_optimizer= optim.Adam(D.parameters(),lr,[beta1,beta2])
g_optimizer= optim.Adam(G.parameters(),lr,[beta1,beta2])


################ training the model on 1 batch of 32 images
d_optimizer.zero_grad()
g_optimizer.zero_grad()
## real Images passed through discriminator
D_real= D(images)
D_real_loss= real_loss(D_real)
## fake images passes throught generator
z= np.random.uniform(-1,1,size=(batch_size,z_size))
z= torch.from_numpy(z).float()

fake_images= G(z)
D_fake= D(fake_images)
D_fake_loss= fake_loss(D_fake)

d_loss= D_real_loss+D_fake_loss
d_loss.backward() 
d_optimizer.step()

##### train the generator
z= np.random.uniform(-1,1,size=(batch_size,z_size))
z= torch.from_numpy(z).float()
G_real = G(z)
D_fake_g= D(G_real)
g_loss= real_loss(D_fake_g)
g_loss.backward()
g_optimizer.step()

#### key take aways --- G_real - to see how the generator function is evolving 



for i in range(100):
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()
    ## real Images passed through discriminator
    D_real= D(images)
    D_real_loss= real_loss(D_real)
    ## fake images passes throught generator
    z= np.random.uniform(-1,1,size=(batch_size,z_size))
    z= torch.from_numpy(z).float()
    fake_images= G(z)
    D_fake= D(fake_images)
    D_fake_loss= fake_loss(D_fake)
    d_loss= D_real_loss+D_fake_loss
    d_loss.backward() 
    d_optimizer.step()
     ##### train the generator
    z= np.random.uniform(-1,1,size=(batch_size,z_size))
    z= torch.from_numpy(z).float()
    G_real = G(z)
    D_fake_g= D(G_real)
    g_loss= real_loss(D_fake_g)
    g_loss.backward()
    g_optimizer.step() 
### print lossese
    print(i,g_loss.item(),d_loss.item())

def plot_generator_(image,plot_size):
    #im= image.detach().numpy()
    if type(image)==torch.Tensor:
        image= image.detach().numpy()
    fig= plt.figure(figsize=(25,4))
    #im=np.transpose(image[pic_number],(1,2,0))
    for idx in np.arange(plot_size):
        ax= fig.add_subplot(2,plot_size/2,idx+1,xticks=[],yticks=[])
        ax= ax.imshow(np.transpose(image[idx],(1,2,0)))
    return ax

#G_real.size() after 100 epochs
#G_out_one= G_real
gg= rev_scale(G_out_one,resize=(1,-1))
plot_generator_(gg,10)

#G_out_two= G_real #after 200 epochs
gg= rev_scale(G_out_two,resize=(1,-1))
plot_generator_(gg,10)

#G_out_three= G_real# after 200 epochs
gg= rev_scale(G_out_three,resize=(1,-1))
plot_generator_(gg,10)

#G_out_four= G_real
gg= rev_scale(G_out_four,resize=(1,-1))
plot_generator_(gg,10)




























