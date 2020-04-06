from config import *
from utils import *
import torch.optim as optim
from torch.autograd import Variable

def inv_map(generator, discriminator, input_image, n_backprop=500, lambd = 0.1,
            device=device, lr=learning_rate, beta1=beta1):
    b_size = input_image.size(0)
    z_latent = noise_data(b_size, n_latent_vector)
    optimizer_z = optim.Adam([z_latent], lr=lr, betas=(beta1, 0.999))
    input_image = input_image.to(device)
    z_latent = z_latent.to(device)
    discriminator_loss = Variable(torch.zeros(b_size,).to(device))
    residual_loss = Variable(torch.zeros(b_size,).to(device))
    total_loss = Variable(torch.zeros(b_size,).to(device))
    for _ in range(n_backprop):
        if z_latent.grad is not None:
            z_latent.grad.zero_()
        generator_z = generator(z_latent)
        _, x_features = discriminator(input_image)
        _, generator_z_features = discriminator(generator_z)
        x_features = x_features.view(b_size, -1)
        generator_z_features = generator_z_features.view(b_size, -1)
        residual_loss = torch.sum(torch.abs(input_image - generator_z))
        discriminator_loss = torch.sum(torch.abs(x_features - generator_z_features))
        total_loss = (1 - lambd)*residual_loss + lambd*discriminator_loss
        total_loss.backward()
        optimizer_z.step()
    return z_latent, total_loss, discriminator_loss, residual_loss
    
    
    
    