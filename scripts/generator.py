from utils import *
from config import *
from typing import Any
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, ngpu: int) -> None:
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.convT1 = deconv(n_latent_vector, n_generator_feature_map * 8, 4, 1, 0)
        self.convT2 = deconv(n_generator_feature_map * 8, n_generator_feature_map * 4, 4, 2, 1)
        self.convT3 = deconv(n_generator_feature_map * 4, n_generator_feature_map * 2, 4, 2, 1)
        self.convT4 = deconv(n_generator_feature_map * 2, n_generator_feature_map, 4, 2, 1)
        self.output = deconv(n_generator_feature_map, n_channels, 4, 2, 1, batch_norm=False)
    
    def forward(self, input):
        out = F.relu(self.convT1(input), inplace=True)
        out = F.relu(self.convT2(out), inplace=True)
        out = F.relu(self.convT3(out), inplace=True)
        out = F.relu(self.convT4(out), inplace=True)
        return torch.tanh(self.output(out))


def train_generator(discriminator, data, label, criterion):
    output = discriminator(data)[0].view(-1)
    error = criterion(output, label)
    error.backward()
    return error, output.mean().item()

# create the generator
generator_network = Generator(ngpu).to(device)

# Handle multi-gpu
if (device.type == 'cuda') and (ngpu > 1):
    generator_network = nn.DataParallel(generator_network, list(range(ngpu)))

generator_network.apply(weights_init)

if __name__ == '__main__':
    print(generator_network)

        
        
    
    
    


