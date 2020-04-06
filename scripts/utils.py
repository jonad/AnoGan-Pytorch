from torch import nn
import os
import torch
from config import  *
def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True, init_zero_weights=False):
    '''Creates a convolutional layer, with optional batch normalization'''
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels,in_channels, kernel_size, kernel_size )*0.001
    layers.append(conv_layer)
    
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True, bias=False):
    '''Creates a convolutional layer, with optional batch normalization'''
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
 
def checkpoint(iteration, generator, discriminator):
  generator_path = os.path.join('models', f'generator{iteration}.pkl')
  discriminator_path = os.path.join('models', f'discriminator{iteration}.pkl')
  torch.save(generator.state_dict(), generator_path)
  torch.save(discriminator.state_dict(), discriminator_path)
  
def get_noise_sampler():
    return lambda m, n: torch.rand(m, n, 1, 1).requires_grad_()

def freeze_network(network):
 for param in network.parameters():
  param.requires_grad = False

def load_checkpoint(model, checkpoint_name):
    model.load_state_dict(torch.load(os.path.join('models', checkpoint_name)))

noise_data = get_noise_sampler()