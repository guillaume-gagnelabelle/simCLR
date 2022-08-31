import torchvision.utils
from torchvision import datasets, transforms
import torch
import numpy as np
from sklearn import datasets as sk_datasets
from util_repr import device
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import numpy as np


def normalize_2D(xs):
    print("normalizing 2D data")
    assert (len(xs.shape) == 2 and xs.shape[1] == 2)
    mean = xs.mean(dim=0, keepdim=True)
    std = xs.std(dim=0, keepdim=True)
    return (xs - mean) / std


def get_data(args):
    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.savefig('data_sample.png')

    if args.data == "MNIST":
        mnist_transforms1 = transforms.Compose([
            transforms.RandomResizedCrop(size=(28, 28)),
            transforms.ToTensor()
        ])

        mnist_transforms2 = transforms.Compose([
            transforms.ColorJitter(brightness=.5, hue=.3),
            transforms.ToTensor()
        ])

        train_data_orig = datasets.MNIST("./MNIST", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
        aug_data1 = datasets.MNIST("./MNIST", train=True, transform=mnist_transforms1, target_transform=None, download=True)
        aug_data2 = datasets.MNIST("./MNIST", train=True, transform=mnist_transforms2, target_transform=None, download=True)
        test_data = datasets.MNIST("./MNIST", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

        train_data_full = []
        for i in range(len(train_data_orig)):
            train_data_full.append([train_data_orig[i], aug_data1[i], aug_data2[i]])

        args.C = 10
        args.in_dim = (1, 28, 28)

    elif args.data == "CIFAR10":
        cifar_transforms1 = transforms.Compose([
            transforms.RandomResizedCrop(size=(32, 32)),
            transforms.ToTensor()
        ])
        cifar_transforms2 = transforms.Compose([
            transforms.ColorJitter(brightness=.5, hue=.3),
            transforms.ToTensor()
        ])

        train_data_orig = datasets.CIFAR10("./CIFAR10", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
        aug_data1 = datasets.CIFAR10("./CIFAR10", train=True, transform=cifar_transforms1, target_transform=None, download=True)
        aug_data2 = datasets.CIFAR10("./CIFAR10", train=True, transform=cifar_transforms2, target_transform=None, download=True)
        test_data = datasets.CIFAR10("./CIFAR10", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

        train_data_full = []
        for i in range(len(train_data_orig)):
            train_data_full.append([train_data_orig[i], aug_data1[i], aug_data2[i]])

        args.C = 10
        args.in_dim = (3, 32, 32)

    if args.visualize_data:
        nb_imgs = 8
        ids = torch.randint(0, len(train_data_full), [nb_imgs]).tolist()

        images, labels = [], []
        for i in ids:
            images.append(train_data_full[i][0][0])
            labels.append(train_data_full[i][0][1])
            images.append(train_data_full[i][1][0])
            labels.append(train_data_full[i][1][1])
            images.append(train_data_full[i][2][0])
            labels.append(train_data_full[i][2][1])

        for i in range(nb_imgs * 3): print(labels[i], end=' ')
        print('')
        imshow(torchvision.utils.make_grid(images))

    num_val = int(len(train_data_full) * args.val_pc)
    val_data, train_data = torch.utils.data.random_split(train_data_full, [num_val, len(train_data_full) - num_val])

    num_val_train = int(0.85 * num_val)
    val_train_data, val_test_data = torch.utils.data.random_split(val_data, [num_val_train, len(val_data) - num_val_train])  # [val_train_data] = [num_val_train, 3 (raw, aug1, aug2), 2 (img, label)]

    # Oh boy
    # Don't want aug. data in val. sets
    val_train_imgs = torch.stack([torch.stack([val_train_data[i][j][0] for j in range(len(val_train_data[i]))]) for i in range(len(val_train_data))]).to(device)  # [num_train_val, 3, chann., H, W]
    val_train_labels = torch.stack([torch.tensor([val_train_data[i][j][1] for j in range(len(val_train_data[i]))]) for i in range(len(val_train_data))]).to(device)  # [num_train_val, 3]
    val_test_imgs = torch.stack([torch.stack([val_test_data[i][j][0] for j in range(len(val_test_data[i]))]) for i in range(len(val_test_data))]).to(device)[:, 0]  # [num_test_val, 3, chann., H, W]
    val_test_labels = torch.stack([torch.tensor([val_test_data[i][j][1] for j in range(len(val_test_data[i]))]) for i in range(len(val_test_data))]).to(device)[:, 0]  # [num_test_val, 3]

    train_imgs = torch.stack([torch.stack([train_data[i][j][0] for j in range(len(train_data[i]))]) for i in range(len(train_data))]).to(device)  # [len()-num_val, 3, 1, H, W]
    # train_labels = torch.stack([torch.tensor([train_data[i][j][1] for j in range(len(train_data[i]))]) for i in range(len(train_data))]).to(device)  # [len()-num_val, 3]

    test_imgs = torch.stack([test_data[i][0] for i in range(len(test_data))]).to(device)  # [len()-num_val, chann., H, W]
    test_labels = torch.tensor([test_data[i][1] for i in range(len(test_data))]).to(device)  # [len()-num_val]

    # train_dataloader_all = torch.utils.data.DataLoader(train_data_full, batch_size=args.batch_sz, shuffle=True)
    # train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_sz, shuffle=True)
    val_train_dataloader = torch.utils.data.DataLoader(val_train_data, batch_size=args.batch_sz, shuffle=True)
    val_test_dataloader = torch.utils.data.DataLoader(val_test_data, batch_size=args.batch_sz, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_sz, shuffle=False)

    return train_imgs, val_train_imgs, val_train_labels, val_train_dataloader,val_test_imgs, val_test_labels, val_test_dataloader, test_imgs, test_labels, test_dataloader
