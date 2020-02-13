from .utils import *
from .config import *
from typing import Any
import torch.nn.functional as F



class Generator(nn.Module):
    def __init__(self, ngpu:int)-> None:
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.convT1 = deconv(n_latent_vector, n_generator_feature_map*8, 4, 1, 0)
        self.convT2 = deconv(n_generator_feature_map*8, n_generator_feature_map*4, 4, 2, 1)
        self.convT3 = deconv(n_generator_feature_map*4, n_generator_feature_map*2, 4, 2, 1)
        self.convT4 = deconv(n_generator_feature_map*2, n_generator_feature_map, 4, 2, 1)
        self.output = deconv(n_generator_feature_map, n_channels, 4, 2, 1, batch_norm=False)
        
    def forward(self, *input: Any, **kwargs: Any) -> T_co:
        out = F.relu(self.convT1(input), inplace=True)
        out = F.relu(self.convT2(out), inplace=True)
        out = F.relu(self.convT3(out), inplace=True)
        out = F.relu(self.convT4(out), inplace=True)
        return F.tanh(self.output(out))
    
    
def train_generator(discriminator, data, label, criterion):
    output = discriminator(data).view(-1)
    error = criterion(output, label)
    error.backward()
    return error
    
    
if __name__ == '__main__':
    
    netG = Generator(ngpu).to(device)
    print(netG)

        
        
    
    
    


