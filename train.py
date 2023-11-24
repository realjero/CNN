import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import squeezenet1_0, vgg11

from utils import SqueezeNet, VGG11

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.AutoAugment(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = ImageFolder(root='./data/Training224r', transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

if __name__ == '__main__':
    # Check if CUDA (GPU) is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # Initialize the model, loss function, and optimizer
    #   model = squeezenet1_0(num_classes=4)
    # Pretrained:
    model = SqueezeNet().get_model()
    # model = VGG11().get_model()

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    plt.ion()
    fig, ax = plt.subplots()
    losses = []

    # Training the model
    for epoch in range(10):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # Update the plot every few iterations
            if len(losses) % 10 == 0:
                ax.clear()
                ax.plot(losses, label='Training Loss')
                ax.set_xlabel('Iterations')
                ax.set_ylabel('Loss')
                ax.legend()
                plt.pause(0.1)  # Add a short pause to update the plot

        print(f'{epoch}: Loss: {losses[-1]}')

    print('Finished Training')
    torch.save(model.state_dict(), 'squeeze_rotated.pth')
    plt.ioff()
    plt.show()
