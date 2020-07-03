import torch
from util.nice_func import NICE
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

device = torch.device("cuda")


def compute_mask(model, test_loader):
    xo_ll = []
    z_ll = []
    for r in [1, 5, 10, 15, 20]:
        model.load_state_dict(torch.load('nice_model/model_r_{}.pth'.format(r)))
        model.eval()
        with torch.no_grad():
            xo_l = []
            z_l = []
            for data, target in test_loader:
                inputs, targets = data.to(device), target.to(device)
                z = model.saliency_l.evaluate(inputs)
                xo_l.append(inputs.cpu().numpy())
                z_l.append(z.cpu().numpy())

            xo = np.vstack(xo_l)
            z = np.vstack(z_l)
        xo_ll.append(xo)
        z_ll.append(z)
    return z_ll, xo_ll


def get_mask(r_idx, im_idx, z_ll):
    return z_ll[r_idx][im_idx, 0]


def get_xo(im_idx, xo_ll):
    return xo_ll[0][im_idx, 0]


def plot2(z_ll, xo_ll):
    imid1 = 15
    imid2 = 110

    f, axarr = plt.subplots(2, 6, figsize=(13, 4))

    axarr[0, 0].imshow(get_xo(imid1, xo_ll), cmap='gray')
    axarr[0, 0].set_title('Input')
    axarr[0, 1].imshow(get_mask(0, imid1, z_ll), cmap='Reds')
    axarr[0, 1].set_title('Lambda = 1')
    axarr[0, 2].imshow(get_mask(1, imid1, z_ll), cmap='Reds')
    axarr[0, 2].set_title('Lambda = 5')
    axarr[0, 3].imshow(get_mask(2, imid1, z_ll), cmap='Reds')
    axarr[0, 3].set_title('Lambda = 10')
    axarr[0, 4].imshow(get_mask(3, imid1, z_ll), cmap='Reds')
    axarr[0, 4].set_title('Lambda = 15')
    axarr[0, 5].imshow(get_mask(4, imid1, z_ll), cmap='Reds')
    axarr[0, 5].set_title('Lambda = 20')

    axarr[1, 0].imshow(get_xo(imid2, xo_ll), cmap='gray')
    axarr[1, 0].set_title('Input')
    axarr[1, 1].imshow(get_mask(0, imid2, z_ll), cmap='Reds')
    axarr[1, 2].imshow(get_mask(1, imid2, z_ll), cmap='Reds')
    axarr[1, 3].imshow(get_mask(2, imid2, z_ll), cmap='Reds')
    axarr[1, 4].imshow(get_mask(3, imid2, z_ll), cmap='Reds')
    axarr[1, 5].imshow(get_mask(4, imid2, z_ll), cmap='Reds')

    [axi.set_axis_off() for axi in axarr.ravel()]

    f.savefig('vis/masks.png', dpi=100)

    plt.show()


def main():
    model = NICE().to(device)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            #                        transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=100, shuffle=False)

    z_ll, xo_ll = compute_mask(model, test_loader)

    if not os.path.exists('vis'):
        os.makedirs('vis')

    plot2(z_ll, xo_ll)


if __name__ == '__main__':
    main()
