from .config import *
from .utils import *
class InverseMapping(nn.Module):
    def __init__(self, generator, discriminator):
        super(InverseMapping, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        
    def forward(self, x, input_image):
        generated_img = self.generator(x)
        _, generated_feats = self.discriminator(generated_img)
        _, image_feats = self.discriminator(input_image)
        return generated_img, generated_feats, image_feats