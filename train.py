import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.models import squeezenet1_0, vgg11, SqueezeNet1_0_Weights

from utils import SqueezeNet, VGG11

transform = SqueezeNet1_0_Weights.DEFAULT.transforms()

training_set = ImageFolder(root='./data/Training224r', transform=transform)
# train, valid = random_split(training_set, [int(len(training_set) / 5 * 4), int(len(training_set) / 5)])

training_loader = DataLoader(training_set, batch_size=64, shuffle=True, num_workers=4)
# validation_loader = DataLoader(valid, batch_size=16, shuffle=True, num_workers=4)

if __name__ == '__main__':
    # Check if CUDA (GPU) is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # Initialize the model, loss function, and optimizer
    #   net = squeezenet1_0(num_classes=4)
    # Pretrained:
    net = SqueezeNet().get_model()
    # net = VGG11().get_model()

    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    losses = []
    # validation_losses = []

    # Training the model
    for epoch in range(10):
        for y, label in training_loader:
            # Training
            net.train()
            y, label = y.to(device), label.to(device)
            optimizer.zero_grad()
            y_hat = net(y)
            loss = criterion(y_hat, label)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # with torch.no_grad():
        #     for y, label in validation_loader:
        #         # Validation
        #         net.eval()
        #         y, label = y.to(device), label.to(device)
        #         validation = net(y)
        #         validation_loss = criterion(validation, label)
        #         validation_losses.append(validation_loss.item())

        print(f'{epoch}: Loss: {losses[-1]}')

    print('Finished Training')

    plt.plot(range(len(losses)), losses, label='Training Loss')
    # plt.plot(range(len(validation_losses)), validation_losses, label='Validation Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    torch.save(net.state_dict(), 'squeeze_rotated.pth')
