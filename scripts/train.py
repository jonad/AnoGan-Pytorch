from .generator import *
from .discriminator import *
from .config import *

def train(dataloader, discriminator, generator, optimizer_d, optimizer_g, criterion):
    print('Starting the training loop')
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 1):
            
            # train the discriminator with real data
            discriminator.zero_grad()
            b_size = data[0].size(0)
            label = torch.full((b_size,), 1, device=device)
            error_d_real = train_discriminator(data, label, discriminator, criterion, device)
            
            # train the discriminator with fake data
            noise = torch.randn(b_size, n_channels , 1, 1, device=device)
            fake = generator(noise)
            label.fill_(1)
            error_d_fake = train_discriminator(fake.detach(), label, discriminator, criterion, device)
            
            # total error
            error_discriminator = error_d_fake + error_d_real
            
            # Update the discriminator's parameters
            optimizer_d.step()
            
            # train the generator
            generator.zero_grad()
            label.fill_(1)
            error_generator = train_generator(fake, label, discriminator)
            optimizer_g.step()
            
    
    
        