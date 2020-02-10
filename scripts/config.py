import torch
from typing import TypeVar
image_size = 64
n_channels = 3
n_latent_vector = 100
n_generator_feature_map = 64
n_discriminator_feature_map = 64
ngpu = 1
beta1 = 0.5
num_epochs = 5
learning_rate=0.0002
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
T_co = TypeVar('T_co', covariant=True)