# import argparse
import torch
# some imports to make pylance happy
import torch.utils
import torch.utils.data
import torch.backends
import torch.backends.mps
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
# from torch.optim.lr_scheduler import StepLR

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2, None, 0, 1, True)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)  # Compressed representation

        # Decoder
        self.fc2 = nn.Linear(128, 9216)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 3, 1)
        self.deconv2 = nn.ConvTranspose2d(32, 1, 3, 1)
        self.unpool = nn.MaxUnpool2d(2)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x, indices = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)

        # Decoder
        x = self.fc2(x)
        x = F.relu(x)
        x = x.view(-1, 64, 12, 12)  # Reshape to match the output shape before the fully connected layer in the encoder
        x = self.unpool(x, indices)
        x = self.deconv1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.deconv2(x)
        x = torch.sigmoid(x)  # Use sigmoid to ensure output values are between 0 and 1
        return x


class MyDataset:
    def __init__(self):
        self.downloaded = False
        self.transform = None
        # self.dataset_train = None
        # self.dataset_test = None

    def get(self,kind):
      if (self.transform is None):
        self.transform = transforms.Compose(
          [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

      if (kind == 'train'):
        dataset = datasets.MNIST("../data", train=True, download=not self.downloaded, transform=self.transform)
      elif (kind == 'test'):
        dataset = datasets.MNIST("../data", train=False, download=not self.downloaded, transform=self.transform)
      else:
        raise ValueError('Invalid dataset kind')
      self.downloaded = True
      return dataset



# def get_dataset(kind):
#     # Load the MNIST dataset
#     # transform = transforms.Compose([transforms.ToTensor()])
#     # train_dataset = datasets.MNIST(
#     #     "data", train=True, download=True, transform=transform
#     # )
#     # test_dataset = datasets.MNIST("data", train=False, transform=transform)
#     # return train_dataset, test_dataset


#     transform = transforms.Compose(
#         [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
#     )
#     if (kind == 'train'):
#       dataset = datasets.MNIST("../data", train=True, download=not downloaded, transform=transform)
#     elif (kind == 'test'):
#       dataset = datasets.MNIST("../data", train=True, download=not downloaded, transform=transform)
#     else:
#       raise ValueError('Invalid dataset kind')
#     downloaded = True
#     return dataset
#     #dataset2 = datasets.MNIST("../data", train=False, transform=transform)
