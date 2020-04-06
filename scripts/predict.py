from inverse_mapping import *
from discriminator import *
from generator import *
from dtsets import *

def predict(dataloader, discriminator, generator, outfile):
    outputs = []
    for batch_no, data in enumerate(dataloader, 0):
        _, total_loss, discriminator_loss, residual_loss = inv_map(generator, discriminator,
                                                                   data[0], n_backprop=500, lambd=0.1,
                                                                   device=device, lr=learning_rate, beta1=beta1)
        total_loss_arr = total_loss.data.cpu()
        discriminator_loss_arr = discriminator_loss.data.cpu()
        residual_loss_arr = residual_loss.data.cpu()
        outputs.append([total_loss_arr, discriminator_loss_arr, residual_loss_arr])
    with open(outfile, 'w') as f:
        f.write('total_loss, discriminator_loss,residual_loss' + '\n')
        for item in outputs:
            f.write(str(item[0]) + ',' + str(item[1]) + ',' + str(item[2]) + '\n')
    
        
        
        
        
    
    




if __name__ == '__main__':
    # Load the model's weights
    load_checkpoint(generator_network, 'generator.pkl')
    load_checkpoint(discriminator_network, 'discriminator.pkl')
    
    # freeze the generator and discriminator
    freeze_network(generator_network)
    freeze_network(discriminator_network)
    

    predict(dataloader_test_normal_images, discriminator_network, generator_network, outfile='test_normal.csv')
    predict(dataloader_test_cancer_images, discriminator_network, generator_network, outfile='test_cancer.csv')
    
