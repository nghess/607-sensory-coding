import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import tifffile as tiff
import pandas as pd
import numpy as np
import os

"""
Load Dataset
"""

class TiffDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = tiff.imread(img_name)  # This should read the entire stack

        if self.transform:
            image = self.transform(image)

        # Ensure the image is a float tensor
        if not isinstance(image, torch.FloatTensor):
            image = image.type(torch.FloatTensor)

        # Convert labels to numerical format
        rotation_class = 1 if self.annotations.iloc[index, 1] == 'clockwise' else 0
        input_class_mapping = {'cross': 0, 'tee': 1, 'elbow': 2, 'radius': 3, 'diameter': 4}
        input_class = input_class_mapping[self.annotations.iloc[index, 2]]

        # Combine the labels (e.g., using one-hot encoding for the input class)
        label = torch.tensor([rotation_class, input_class], dtype=torch.long)

        return image, label

# Example usage
transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = TiffDataset(csv_file='dataset/labels.csv', root_dir='dataset/', transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

"""
Define Model
"""

class Simple3DCNN(nn.Module):
    def __init__(self):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv3d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 45 * 64, 512)  # Adjust the size
        self.fc_rotation = nn.Linear(512, 2)  # Two classes for rotation
        self.fc_input = nn.Linear(512, 5)  # Five classes for input type

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.shape) torch.Size([32, 64, 45, 64])
        x = x.view(-1, 32 * 64 * 45 * 64)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        rotation_output = self.fc_rotation(x)
        input_output = self.fc_input(x)
        return rotation_output, input_output

model = Simple3DCNN()

"""
Train Model
"""

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the device (GPU or CPU)
model = model.to(device)

for epoch in range(50):
    for i, (inputs, labels) in enumerate(dataloader):
        
        # Split the labels
        rotation_labels = labels[:, 0]
        input_labels = labels[:, 1]

        optimizer.zero_grad()
        outputs = model(inputs)

        # Assuming your model's output is designed to handle both types of labels
        loss_rotation = criterion(outputs[0], rotation_labels)
        loss_input = criterion(outputs[1], input_labels)
        loss = loss_rotation + loss_input  # Combine losses, or handle as you see fit

        loss.backward()
        optimizer.step()

"""
Test Model
"""
def test_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct_rotation, correct_input, total = 0, 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            rotation_labels, input_labels = labels[:, 0], labels[:, 1]

            rotation_output, input_output = model(inputs)

            _, predicted_rotation = torch.max(rotation_output.data, 1)
            _, predicted_input = torch.max(input_output.data, 1)

            total += labels.size(0)
            correct_rotation += (predicted_rotation == rotation_labels).sum().item()
            correct_input += (predicted_input == input_labels).sum().item()

    print(f'Accuracy of the network on rotation prediction: {100 * correct_rotation / total}%')
    print(f'Accuracy of the network on input type prediction: {100 * correct_input / total}%')