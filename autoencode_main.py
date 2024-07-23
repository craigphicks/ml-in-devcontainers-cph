import argparse
import torch
# some imports to make pylance happy
import torch.utils
import torch.utils.data
import torch.backends
import torch.backends.mps
#import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from autoencoder_net import MyNet, get_dataset

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         # Encoder
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.pool = nn.MaxPool2d(2, None, 0, 1, True)
#         self.dropout1 = nn.Dropout(0.25)
#         self.fc1 = nn.Linear(9216, 128)  # Compressed representation

#         # Decoder
#         self.fc2 = nn.Linear(128, 9216)
#         self.deconv1 = nn.ConvTranspose2d(64, 32, 3, 1)
#         self.deconv2 = nn.ConvTranspose2d(32, 1, 3, 1)
#         self.unpool = nn.MaxUnpool2d(2)
#         self.dropout2 = nn.Dropout(0.5)

#     def forward(self, x):
#         # Encoder
#         # x = F.relu(self.conv1(x))
#         # x = F.relu(self.conv2(x))
#         # x, indices = self.pool(x)
#         # x = self.dropout1(x)
#         # x = torch.flatten(x, 1)
#         # x = F.relu(self.fc1(x))

#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x, indices = self.pool(x)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)

#         # Decoder
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = x.view(-1, 64, 12, 12)  # Reshape to match the output shape before the fully connected layer in the encoder
#         x = self.unpool(x, indices)
#         x = self.deconv1(x)
#         x = F.relu(x)
#         # x = self.dropout2(x)
#         x = self.deconv2(x)
#         x = torch.sigmoid(x)  # Use sigmoid to ensure output values are between 0 and 1
#         return x

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, _) in enumerate(train_loader):  # Target not used
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        #print(f"[DBG] Output shape: {output.shape}, Input shape: {data.shape}")
        loss = F.mse_loss(output, data)  # Changed to MSE loss
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:  # Target not used
            data = data.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, data, reduction='sum').item()  # Changed to MSE loss

    test_loss /= len(test_loader.dataset)

    print("\nTest set: Average loss: {:.4f}\n".format(test_loss))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        default=False,
        help="disables macOS GPU training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # transform = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    # )
    dataset1 = get_dataset('train')
    dataset2 = get_dataset('test')
    # dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    # dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = MyNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
