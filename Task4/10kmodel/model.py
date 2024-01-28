import os
os.environ["KERAS_BACKEND"] = "torch"  # Or "jax" or "torch"!
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import keras
from keras.utils import plot_model

import torch
from torch import nn
#import tqdm
#from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchsummary import summary

from torchvision import datasets, transforms

from torchviz import make_dot

print(torch.cuda.device_count())

device = "cuda" if torch.cuda.is_available() else 'cpu'
print(f'The device we will be working on is: {device}')


IMG_CHANNELS = 3
OUTPUT_SHAPE = 8
N_EPOCHS = 75


class net(nn.Module):

    def __init__(self):
        super().__init__()
        self.dropout_val = 0.5

        self.conv_block_first = nn.Sequential(
            nn.Conv2d(in_channels=IMG_CHANNELS,
                      out_channels=24,
                      kernel_size=2,  
                      stride=1,
                      padding=1),  
            nn.BatchNorm2d(24),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=24,
                      out_channels=24,
                      kernel_size=2,  
                      stride=2,
                      padding=1),  
            nn.BatchNorm2d(24),
            nn.ReLU(),
        
            nn.Conv2d(in_channels=24,
                      out_channels=24,
                      kernel_size=2, 
                      stride=2,
                      padding=1), 
            nn.BatchNorm2d(24),
            nn.ReLU(),
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=24,
                      out_channels=48,
                      kernel_size=2, 
                      stride=1,
                      padding=1),  
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )
        
        self.dense_block = nn.Sequential(
            nn.Flatten()
        )
        
        self.output = nn.Sequential(
            nn.Linear(in_features=48, # ?
                      out_features=OUTPUT_SHAPE)
        )

    def forward(self, x):
        x = self.conv_block_first(x)
        x = self.pool2(x)
        x = self.conv_block1(x)
        x = self.pool2(x)
        x = self.conv_block2(x)
        x = nn.functional.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.dense_block(x)
        x = self.output(x)
        return x

model = net().to(device)
summary(model, (3, 256, 256))

'''
x = torch.randn(16, 3, 256, 256).to(device)
dot = make_dot(model(x), params=dict(model.named_parameters()))
dot.render('/ghome/group07/prueba_torch/prueba_torch', format='png')
'''

# optimizer = torch.optim.Adam(model.parameters())
# optimizer = torch.optim.RMSprop(model.parameters())
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# optimizer = torch.optim.Adagrad(model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
loss_fn = torch.nn.CrossEntropyLoss()


DATASET_DIR = '/ghome/group07/MIT_small_augments/small_bigger_2'
# DATASET_DIR = '/ghome/group07/mcv/datasets/C3/MIT_small_train_1'
batch_size = 8


transform = transforms.Compose([
    #transforms.Resize((256, 256)),  # Resize images to a fixed size
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values
])

train_path = DATASET_DIR+'/train'
test_path = DATASET_DIR+'/test'

train_data = datasets.ImageFolder(root=train_path, transform=transform)
test_data = datasets.ImageFolder(root=test_path, transform=transform)

train_loader = torch.utils.data.DataLoader(
                    dataset=train_data,
                    batch_size=batch_size,
                    shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_data,
                batch_size=batch_size,
                shuffle=False)

for inputs, labels in train_loader:
    inputs, labels = inputs.to(device), labels.to(device)

for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)


def accuracy_fn(y_true, y_pred):
    correct = troch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred))*100
    return acc

def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.

    total_samples = 0
    correct_predictions = 0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        _, predicted = torch.max(outputs, 1)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        running_loss += loss

        loss.backward()

        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        # Adjust learning weights
        optimizer.step()

        '''
        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
        '''
    train_loss = running_loss/total_samples
    train_acc = correct_predictions / total_samples

    return train_loss, train_acc

# Initializing in a separate cell so we can easily add more epochs to the same run
epoch_number = 0

best_tloss = 1_000_000.

def calculate_accuracy_and_loss(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode

    correct_predictions = 0
    total_samples = 0

    loss = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Get the predicted class labels
            _, predicted = torch.max(outputs, 1)

            # Update counts
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            loss += loss_fn(outputs, labels)

    accuracy = correct_predictions / total_samples
    loss /= total_samples
    
    return accuracy, loss

for epoch in range(N_EPOCHS):
    start_time = time.time()
    model.train(True)
    train_loss, train_acc = train_one_epoch(epoch_number)

    model.eval()
    val_acc, val_loss = calculate_accuracy_and_loss(model, test_loader, device=device)

    end_time = time.time()
    epoch_duration = end_time - start_time
    print(f'[Epoch: {epoch + 1}/{N_EPOCHS} ({epoch_duration:.2f}s)] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

    # Track best performance, and save the model's state
    if val_loss < best_tloss:
        best_tloss = val_loss
        model_path = 'dataset2/model5/prueba4/solo2conv_con_init_batch16.pt'
        torch.save(model.state_dict(), model_path)

    epoch_number += 1