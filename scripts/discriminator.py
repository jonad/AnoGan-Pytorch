from .config import *
from .utils import *
from typing import Any
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        
        self.conv1 = conv(n_channels, n_discriminator_feature_map, 4, 2, 1)
        self.conv2 = conv(n_discriminator_feature_map, n_discriminator_feature_map*2, 4, 2, 1)
        self.conv3 = conv(n_discriminator_feature_map*2, n_discriminator_feature_map*4, 4, 2, 1)
        self.conv4 = conv(n_discriminator_feature_map*4, n_discriminator_feature_map*8, 4, 2, 1)
        self.output = conv(n_discriminator_feature_map*8, 1, 4, 1, 0, batch_norm=False)
        
    def forward(self, *input: Any, **kwargs: Any) -> T_co:
        out = F.leaky_relu(self.conv1(input), inplace=True)
        out = F.leaky_relu(self.conv2(out), inplace=True)
        out = F.leaky_relu(self.conv3(out), inplace=True)
        out = F.leaky_relu(self.conv4(out), inplace=True)
        return F.sigmoid(self.output(out)), out.view(-1)
    
if __name__ == '__main__':
    netD = Discriminator(ngpu).to(device)
    print(netD)
    
    