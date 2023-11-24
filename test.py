import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from utils import SqueezeNet, VGG11

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

testset = ImageFolder(root='./data/Test', transform=transform)
testloader = DataLoader(testset, batch_size=1, num_workers=4)

classes = ['cup', 'key', 'pencil', 'scissor']

if __name__ == '__main__':
    # Check if CUDA (GPU) is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    model = SqueezeNet(weights="squeeze_rotated.pth").get_model()
    # model = VGG11(weights='vgg_rotated.pth').get_model()
    model.to(device)
    model.eval()

    correct_predictions = 0
    total_samples = 0
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)

    with torch.no_grad():
        for img, label in testloader:
            img = img.to(device)
            output = model(img)

            probabilities = torch.nn.functional.softmax(output[0], dim=0)

            # Get the predicted class
            _, predicted_class = torch.max(output, 1)

            total_samples += 1
            if predicted_class == label.to(device):
                correct_predictions += 1
                class_correct[label] += 1

            class_total[label] += 1

    print(f"Amount: {len(testloader)}")

    accuracy = correct_predictions / total_samples
    print(f"Accuracy: {accuracy}")

    for i in range(len(classes)):
        class_accuracy = class_correct[i] / class_total[i] if class_total[i] != 0 else 0
        print(f"Accuracy for {classes[i]}: {class_accuracy}")

    precision = sum(class_correct) / sum(class_total) if sum(class_total) != 0 else 0
    recall = accuracy
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")
