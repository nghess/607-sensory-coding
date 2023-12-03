import matplotlib.pyplot as plt
import torch
import torchvision.models as models

# Load a pre-trained model, for example, VGG16
model = models.vgg16(pretrained=True)

# Assuming we want to visualize filters from the first convolutional layer
first_conv_layer = model.features[0]

# Extract the filters
filters = first_conv_layer.weight.data.clone()

# Normalize filter values for visualization
filters = (filters - filters.min()) / (filters.max() - filters.min())

# Plot some of the filters
plt.figure(figsize=(20, 17))
for i, filter in enumerate(filters[:16]):  # Visualize first 16 filters
    plt.subplot(4, 4, i+1)  # 4x4 grid
    plt.imshow(filter.permute(1, 2, 0))  # Permute the tensor to image format
    plt.axis('off')
plt.show()