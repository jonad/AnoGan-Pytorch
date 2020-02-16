from .config import *
from .utils import *
import torch.optim as optim

def inv_map(generator, discriminator, input_image, n_backprop=500, lambd = 0.1,
            device=device, lr=learning_rate, beta1=beta1):
    z_latent = get_noise_sampler(1, n_latent_vector)
    optimizer_z = optim.Adam([z_latent], lr=lr, betas=(beta1, 0.999))
    z_latent = z_latent.to(device)
    total_loss = 0
    for _ in range(n_backprop):
        if z_latent.grad is not None:
            z_latent.grad.zero_()
        generator_z = generator(z_latent)
        _, x_features = discriminator(input_image)
        _, generator_z_features = discriminator(generator_z)
        residual_loss = torch.sum(torch.abs(input_image - generator_z))
        discriminator_loss = torch.sum(torch.abs(x_features - generator_z_features))
        total_loss = (1 - lambd)*residual_loss + lambd*discriminator_loss
        total_loss.backward()
        optimizer_z.step()
    return z_latent, total_loss
    
    
    
    