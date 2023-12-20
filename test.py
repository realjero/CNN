import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from torchvision.models import SqueezeNet1_0_Weights

from utils import SqueezeNet

transform = SqueezeNet1_0_Weights.DEFAULT.transforms()

testset = ImageFolder(root='./data/bonustest2', transform=transform)
testloader = DataLoader(testset, batch_size=1, num_workers=4)

classes = ['cup', 'key', 'pencil', 'scissor']

if __name__ == '__main__':
    # Check if CUDA (GPU) is available
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    net = SqueezeNet(weights="squeeze_rotated.pth").get_model()
    net.to(device)
    net.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for y, label in testloader:
            y = y.to(device)
            y_hat = net(y)

            probabilities = torch.nn.functional.softmax(y_hat[0], dim=0)

            # Get the predicted class
            _, predicted_class = torch.max(y_hat, 1)

            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(predicted_class.cpu().numpy())

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    confusion_mat = confusion_matrix(all_labels, all_predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=classes)
    cm_display.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.show()
