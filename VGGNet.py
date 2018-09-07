import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Device configuration
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    device=torch.device('cuda:0')
    print("Using GPU")
else:
    device=torch.device('cpu')
    print("Using CPU")

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 256
learning_rate = 1e-2
dropout=0.5
L2wd=5e-4
sgdmom=0.9

'''[32x32x3] INPUT
[32x32x64] Conv3-64-1
[32x32x64] Conv3-64-1
[16x16x64] Pool-2-0
[16x16x128] Conv3-128-1
[16x16x128] Conv3-128-1
[8x8x128] Pool-2-0
[8x8x256] Conv3-256-
[8x8x256] Conv3-256-
[8x8x256] Conv3-256-
[4x4x256] Pool-2-
[4x4x512] Conv3-512-
[4x4x512] Conv3-512-
[4x4x512] Conv3-512-
[2x2x512] Pool-2-
[2x2x512] Conv3-512-
[2x2x512] Conv3-512-
[2x2x512] Conv3-512-
[1x1x512] Pool-2
FC 4096
FC 4096
FC 1000

Conv stride: 1
Max pool stride:2'''

# CIFAR10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='/media/sohaib/DATA/NUST/TUKL/Data/',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='/media/sohaib/DATA/NUST/TUKL/Data/',
                                            train=False,
                                            transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Convolutional neural network (two convolutional layers)
class VGGNet(nn.Module):
    def __init__(self, num_classes=10):
        super(VGGNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 4096),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        out = self.layer1(x)
        out=out.view(out.size(0), -1)
        out = self.layer2(out)
        return out

model = VGGNet(num_classes).to(device)

#model = torch.load('Vmodel.pt')

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=sgdmom, weight_decay=L2wd)

# Train the model
total_step = len(train_loader)
print("Training Now")
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the training images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model, 'Vmodel.pt')