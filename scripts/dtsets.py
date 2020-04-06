from torchvision import datasets
from torchvision import transforms
import torch.utils.data
from config import *


training_images = datasets.ImageFolder(root=TRAINING_FOLDER,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5],[0.5]),
                            ]))

# test_normal_images = datasets.ImageFolder(root=TEST_FOLDER_NORMAL,
#                             transform=transforms.Compose([
#                                 transforms.Resize(image_size),
#                                 transforms.CenterCrop(image_size),
#                                 transforms.Grayscale(num_output_channels=1),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize([0.5], [0.5]),
#                             ]))

# test_cancer_images = datasets.ImageFolder(root=TEST_FOLDER_CANCER,
#                             transform=transforms.Compose([
#                                 transforms.Resize(image_size),
#                                 transforms.CenterCrop(image_size),
#                                 transforms.Grayscale(num_output_channels=1),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize([0.5], [0.5]),
#                             ]))

dataloader_training_images = torch.utils.data.DataLoader(training_images, batch_size=batch_size,
                                                          shuffle=True, num_workers=num_workers)
# dataloader_test_cancer_images = torch.utils.data.DataLoader(test_cancer_images, batch_size=1,
#                                                              shuffle=True, num_workers=num_workers)
# dataloader_test_normal_images = torch.utils.data.DataLoader(test_normal_images, batch_size=1,
#                                                              shuffle=True, num_workers=num_workers)