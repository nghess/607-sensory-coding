import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import tifffile as tiff
import pandas as pd
import os


"""
Data Wrangling
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
        image = tiff.imread(img_name)  # Load stack

        # Manually convert the image to a tensor and normalize
        image = torch.from_numpy(image).type(torch.FloatTensor) / 255.0

        # Add channel dim
        image = image.unsqueeze(0)
        image = image.permute(1, 0, 2, 3)

        # Print the shape of the image
        # print(image.shape)

        # Apply any transformations
        if self.transform:
            image = self.transform(image)

        # Ensure the image is a float tensor
        if not isinstance(image, torch.FloatTensor):
            image = image.type(torch.FloatTensor)

        # Convert labels to numerical format
        rotation_class = 1 if self.annotations.iloc[index, 1] == 'clockwise' else 0
        angle_class = int(self.annotations.iloc[index, 2])

        # Combine the labels (e.g., using one-hot encoding for the input class)
        label = torch.tensor([rotation_class, angle_class], dtype=torch.long)

        return image, label

dataset = TiffDataset(csv_file='dataset/slice/labels_slice.csv', root_dir='dataset/')

# Determine the lengths for train and test sets
train_size = int(0.6 * len(dataset))
test_size = len(dataset) - train_size

# Split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

"""
Define Model
"""
# Input/Output Arguments
conv1_in_ch = 1
conv1_out_ch = 4
conv2_in_ch = conv1_out_ch
conv2_out_ch = 8
im_fc_in = conv2_out_ch * 128 * 128 # Channels * input height * input width
fc_in = conv2_out_ch * 128 * 128 * 5
im_fc_out = 512
lstm_out = 512

class ConvLSTMNetwork(nn.Module):
    def __init__(self, num_frames, num_classes_rotation, num_classes_angle):
        super(ConvLSTMNetwork, self).__init__()

        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, padding=2)
        
        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, padding=2)

        # Third Convolutional Layer
        #self.conv3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=2)

        # Dimensionality reduction layer
        self.intermediate_fc = nn.Linear(im_fc_in, im_fc_out)

        # Fully Connected Layers for classification
        self.fc_rotation = nn.Linear(fc_in, num_classes_rotation)
        self.fc_angle = nn.Linear(fc_in, num_classes_angle)

        # Activation Functions
        self.relu = nn.ReLU()

    def forward(self, x):
        # batch_size=4, num_frames=5, channels=1, height=128, width=128
        batch_size, num_frames, channels, height, width = x.size()

        # Process each frame through convolutional layers
        conv_outputs = []
        for t in range(num_frames):
            # Extracting frame t from each stack in the batch
            frame = x[:, t, :, :, :]
            
            # Pass the frame through convolutional layers
            out = self.relu(self.conv1(frame))
            out = self.relu(self.conv2(out))
            #out = self.relu(self.conv3(out))
            
            # Flatten the output for FC layers
            out = out.view(batch_size, -1) 
            #print(f"conv2 output:{out.shape}")

            # Append output to list
            conv_outputs.append(out)

        # Concatenate the outputs along a specific dimension (e.g., 1)
        x = torch.cat(conv_outputs, dim=1)

        # Fully Connected Layers for classification
        output_rotation = self.fc_rotation(x)
        output_angle = self.fc_angle(x)

        return output_rotation, output_angle
        
# Dataset Params
num_frames = 5  # Number of frames in each stack
num_classes_rotation = 2  # Number of output classes for rotation_class
num_classes_angle = 36  # Number of output classes for angle_class

# Initialize model
model = ConvLSTMNetwork(num_frames, num_classes_rotation, num_classes_angle)

# Define loss functions for both classification tasks
criterion_rotation = nn.CrossEntropyLoss()
criterion_angle = nn.CrossEntropyLoss()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)


"""
Check CUDA and load model to GPU
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if torch.cuda.is_available():
    print(device)
    print("GPU enabled!")
else:
    print(device)
    print("GPU **not** enabled!")   


"""
Train Model
"""

# Training loop
num_epochs = 300
running_loss = []

for epoch in range(num_epochs):  # num_epochs is the number of epochs
    model.train()  # Set the model to training mode
    total_loss = 0

    # Schedule learning rate optimizer
    if epoch >= num_epochs/10:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.0005)

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to the same device as the model

        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        output_rotation, output_angle = model(inputs)

        # Separate the labels for rotation and angle
        labels_rotation = labels[:, 0]
        labels_angle = labels[:, 1]

        # Calculate losses for each task
        loss_rotation = criterion_rotation(output_rotation, labels_rotation)
        loss_angle = criterion_angle(output_angle, labels_angle)

        # Combine the losses
        loss = (loss_rotation/2) + (loss_angle)

        # Backward pass and optimization
        loss.backward()  # Compute the gradients
        optimizer.step()  # Update the parameters

        total_loss += loss.item()

    # Print average loss for the epoch
    average_loss = total_loss / len(train_loader)
    running_loss.append(average_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}")


"""
Save Model
"""
model_name = "looped_2dcnn_3layer_final_attempt"
torch.save(model, f"{model_name}.pt")

with open(f"{model_name}.txt", 'a') as file:
    # Iterate over the list
    for item in running_loss:
        # Write each item to the file followed by a newline
        file.write(f"{item},\n")


"""
Test
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

test_model(model, test_loader, device)