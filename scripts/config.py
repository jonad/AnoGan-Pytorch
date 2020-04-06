import torch
import random

seed = 100
random.seed(seed)

TRAINING_FOLDER = '/training_data'
TEST_FOLDER_NORMAL = '/normal'
TEST_FOLDER_CANCER = '/cancer'
CHECKPOINT_DIR = '/models/'



image_size = 64
n_channels = 1
n_latent_vector = 100
n_generator_feature_map = 64
n_discriminator_feature_map = 64
ngpu = 1
beta1 = 0.5
num_epochs = 20
learning_rate=0.0002
batch_size = 64
num_workers=2
real_label = 1
fake_label = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


