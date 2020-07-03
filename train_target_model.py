from torch.nn import functional as F
import numpy as np
import torch.nn as nn
import torch
import os
from torchvision import datasets, transforms
import torch.optim.lr_scheduler as lrsch
from util.models import Net
import os

device = torch.device("cuda")

# load data
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       # transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=100, shuffle=True)

# load model
model = Net().to(device)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = lrsch.StepLR(optimizer, step_size=5, gamma=0.1)

# Train the model
for epoch in range(10):
    scheduler.step()
    model.train()
    for batch_idx, (xb, yb) in enumerate(train_loader):
        # Convert numpy arrays to torch tensors
        inputs = xb.to(device)
        targets = yb.to(device)

        # Forward pass
        outputs = model(inputs)

        loss = F.nll_loss(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 1 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, 10, loss.item()))

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if not os.path.exists('pretrained_model'):
    os.makedirs('pretrained_model')
torch.save(model.state_dict(), 'pretrained_model/model.pth')
