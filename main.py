from torch.nn import functional as F
import numpy as np
import torch.nn as nn
import torch
import os
from torchvision import datasets, transforms
import torch.optim.lr_scheduler as lrsch
from util.nice_func import NICE
import argparse

def main(args):
    device = torch.device("cuda")
    r = args.r

    # load data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=100, shuffle=True)

    # creat NICE model
    model = NICE().to(device)
    model.cnn.load_state_dict(torch.load('pretrained_model/model.pth'))

    # freeze target model
    for p in model.cnn.parameters():
        p.requires_grad = False

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = lrsch.StepLR(optimizer, step_size=5, gamma=0.1)

    # Train the model
    for epoch in range(10):

        model.train()
        for batch_idx, (xb, yb) in enumerate(train_loader):
            # Convert numpy arrays to torch tensors
            inputs = xb.to(device)
            targets = yb.to(device)

            # Forward pass
            outputs, z, loss2 = model(inputs)

            loss1 = F.nll_loss(outputs, targets)
            loss2 = torch.mean(loss2)
            loss = loss1 + r * loss2

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 1 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}'.format(epoch + 1, 10, loss.item(),
                                                                                     loss1.item(), loss2.item() * r))

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output, _, _ = model(data)
                test_loss += F.nll_loss(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    if not os.path.exists('nice_model'):
        os.makedirs('nice_model')
    torch.save(model.state_dict(), 'nice_model/model_r_{}.pth'.format(r))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--r', default=1, type=int)
    args = parser.parse_args()
    main(args)
