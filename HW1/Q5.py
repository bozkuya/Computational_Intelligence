#import libraries
import torchvision
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
# Set parameters
BATCH_SIZE = 50
EPOCH_SIZE = 20
# I run the application on CPU
device = torch.device('cpu')
# Define a custom transformation to be applied to the images
custom_transform = transforms.Compose([
transforms.ToTensor(),
# normalize image values
transforms.Normalize((0.5,), (0.5,))
])
# Load the Fashion MNIST dataset and apply the custom transformation
train_data = torchvision.datasets.FashionMNIST(
'./data', train=True, download=True, transform=custom_transform
)
train_set, val_set = train_test_split(train_data, test_size=0.1, 
random_state=42)
#class for CNN
class CNN3(nn.Module):
def __init__(self, num_classes):
super(CNN3, self).__init__()
self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, 
stride=1, padding='valid')
self.relu1 = nn.ReLU()
self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=7, 
stride=1, padding='valid')
self.relu2 = nn.ReLU()
self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, 
stride=1, padding='valid')
self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0)
self.fc1 = nn.Linear(in_features=144, out_features=num_classes) 
def forward(self, x):
x = self.conv1(x)
2304202
x = self.relu1(x)
x = self.conv2(x)
x = self.relu2(x)
x = self.maxpool1(x)
x = self.conv3(x)
x = self.maxpool2(x)
x = x.view(x.size(0), -1)
x = self.fc1(x)
return x
train_losses_total = []
valid_accus_total = []
lrs = [0.1, 0.01, 0.001]
for lr in lrs:
print(f"rate {lr}")
model = CNN3(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
train_losses = []
valid_accus = []
for epoch in range(EPOCH_SIZE):
print(f"Epoch is {epoch+1}/{EPOCH_SIZE}")
total, correct = 0, 0
train_generator = torch.utils.data.DataLoader(train_set, 
batch_size=BATCH_SIZE, shuffle=True)
total_step = len(train_generator)
for i, (images, labels) in enumerate(train_generator): 
model.train()
images, labels = images.to(device), labels.to(device)
optimizer.zero_grad()
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
if (i+1) % 10 == 0:
model.eval()
with torch.no_grad():
running_loss = loss.item()
val_generator = torch.utils.data.DataLoader(val_set, 
batch_size=BATCH_SIZE, shuffle=False)
val_total, val_correct = 0, 0
for j, (val_images, val_labels) in
enumerate(val_generator): 

2304202
val_images, val_labels = val_images.to(device), 
val_labels.to(device)
val_images = val_images.to(device)
val_labels = val_labels.to(device)
# Forward pass
val_outputs = model_mlp(val_images)
_, val_predicted = val_outputs.max(1)
val_total += val_labels.size(0)
val_correct += val_predicted.eq(val_labels).sum().item()
val_accu=(val_correct/ val_total)*100
train_losses.append(running_loss)
valid_accus.append(val_accu)
train_losses_total.append(train_losses)
valid_accus_total.append(valid_accus)
#Dictionary for json
dictonary ={
'name': 'cnn3',
'loss_curve_1':train_losses_total[0],
'loss_curve_01':train_losses_total[1],
'loss_curve_001':train_losses_total[2],
'val_acc_curve_1':valid_accus_total[0],
'val_acc_curve_01':valid_accus_total[1], 
'val_acc_curve_001':valid_accus_total[2],
}
#Recording Results
with open("Q5cnn3.json","w") as outfile:
json.dump(dictonary,outfile)
#Yasincan Bozkurt 