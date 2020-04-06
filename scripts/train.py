from generator import *
from discriminator import *
from config import *
from dtsets import *
import torch.optim as optim

def train(dataloader, discriminator, generator, optimizer_d, optimizer_g, criterion, device):
    
    print('Starting the training loop ...')
    for epoch in range(num_epochs):
        generator_losses = []
        discriminator_losses = []
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
            generator_losses.append(error_generator.item())
            discriminator_losses.append(error_discriminator.item())
            
        print(f'{epoch:<10} {"generator loss:"} {sum(generator_losses)/len(generator_losses)}\
        {"discriminator loss:"} {sum(discriminator_losses)/len(discriminator_losses)}')
        
        checkpoint(epoch, generator_network, discriminator_network)

if __name__ == '__main__':
    criterion = nn.BCELoss()
    create_dir('models')
    
    optimizer_D = optim.Adam(discriminator_network.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizer_G = optim.Adam(generator_network.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    train(dataloader_training_images, discriminator_network, generator_network, optimizer_D, optimizer_G, criterion, device)
    

            
            
    
    
        