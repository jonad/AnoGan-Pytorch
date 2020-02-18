from .generator import *
from .discriminator import *
from .config import *
from .datasets import *
import torch.optim as optim

def train(dataloader, discriminator, generator, optimizer_d, optimizer_g, criterion, device):
    generator_losses = []
    discriminator_losses = []
    print('Starting the training loop')
    print(f'{"Epoch":<10} {"Batch":<10} {"Discriminator Loss":<20} '\
          f'{"Generator Loss":<20} {"Discriminator(x)":<20}'\
          f' {"Discriminator(Generator(z))":<20}')
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 1):
            
            # train the discriminator with real data
            discriminator.zero_grad()
            b_size = data[0].size(0)
            label = torch.full((b_size,), real_label, device=device)
            error_d_real, output_d_real = train_discriminator(data[0], label, discriminator, criterion, device)
            
            # train the discriminator with fake data
            noise = noise_data(b_size, n_latent_vector).to(device)
            fake = generator(noise)
            label.fill_(fake_label)
            error_d_fake, output_d_fake = train_discriminator(fake.detach(), label, discriminator, criterion, device)
            
            # total error
            error_discriminator = error_d_fake + error_d_real
            
            # Update the discriminator's parameters
            optimizer_d.step()
            
            # train the generator
            generator.zero_grad()
            label.fill_(real_label)
            error_generator, output_g_fake = train_generator(discriminator, fake, label, criterion)
            optimizer_g.step()
            if i % batch_size == 0:
                print(f'{epoch:<10}{i:<10}{error_discriminator.item():<10.4f}\
                            {error_generator.item():<5.4f}\
                              {output_d_real:<5.4f}\
                              {output_d_fake:<5.4f} / {output_g_fake:.4f}')
    
            generator_losses.append(error_generator.item())
            discriminator_losses.append(error_discriminator.item())
    return generator_losses, discriminator_losses

if __name__ == '__main__':
    criterion = nn.BCELoss()
    
    optimizer_D = optim.Adam(discriminator_network.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizer_G = optim.Adam(generator_network.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    _, _ = train(dataloader_training_images, discriminator_network, generator_network, optimizer_D, optimizer_G, criterion, device)
    create_dir(MODEL_FOLDER)
    checkpoint(generator_network, discriminator_network)
            
            
    
    
        