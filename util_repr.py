import torch
import torch.nn as nn
import numpy as np
from models_simCLR import ClassifierModel
from sklearn.linear_model import LogisticRegression
from copy import deepcopy
from generic import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def logistic_reg(args, enc, val_train_imgs, val_train_labels, val_test_imgs, val_test_labels, t):
    enc.eval()
    classifier = LogisticRegression()
    # xent = nn.CrossEntropyLoss(reduction='sum')

    x_train = enc(val_train_imgs).cpu().detach().numpy()
    y_train = val_train_labels.cpu().detach().numpy()

    classifier.fit(x_train, y_train)
    assert (np.sort(classifier.classes_) == np.arange(args.C)).all()
    
    x_test = enc(val_test_imgs).cpu().detach().numpy()
    y_test = val_test_labels
    distr_test = torch.from_numpy(classifier.predict_proba(x_test)).to(device)
    pred_test = torch.from_numpy(classifier.predict(x_test)).to(device)
    distr_train = torch.from_numpy(classifier.predict_proba(x_train)).to(device)
    pred_train = torch.from_numpy(classifier.predict(x_train)).to(device)

    assert distr_test.shape == (len(x_test), args.C)
    assert pred_test.shape == (len(x_test),)
    assert distr_train.shape == (len(x_train), args.C)
    assert pred_train.shape == (len(x_train),)

    correct_train = pred_train.eq(val_train_labels).sum().item()
    total_train = len(x_train)
    acc_train = 100. * correct_train / total_train
    avg_loss_train = xent(distr_train, val_train_labels, reduction='sum').item() / total_train

    args.logs['val_train_loss'][t] = avg_loss_train
    args.logs['val_train_acc'][t] = acc_train

    correct_test = pred_test.eq(y_test).sum().item()
    total_test = len(x_test)
    acc_test = 100. * correct_test / total_test
    avg_loss_test = xent(distr_test, y_test, reduction='sum').item() / total_test

    args.logs['val_test_loss'][t] = avg_loss_test
    args.logs['val_test_acc'][t] = acc_test

    # if args.best_acc < acc:
    #     args.best_acc = acc
    #     args.best_timestep = t

    if args.best_loss > avg_loss_test:
        args.best_loss = avg_loss_test
        args.best_timestep = t

    enc.train()
    return classifier


def evaluation_test_linlayer(args, enc, mlp, test_loader, t):
    enc.eval()
    mlp.eval()
    # xent = nn.CrossEntropyLoss(reduction='sum')

    # Test set
    loss = 0.
    correct = 0.
    total = 0.
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)

            _, pred = mlp(enc(images)).softmax(1).max(1)
            loss += xent(mlp(enc(images)).softmax(1), labels, reduction='sum').item()
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
    avg_loss = loss / total
    acc = 100. * correct / total

    args.logs['test_loss'][t] = avg_loss
    args.logs['test_acc'][t] = acc

    enc.train()
    mlp.train()
    return 0

def evaluation_valtrain_linlayer(args, enc, mlp, val_train_loader, t):
    enc.eval()
    mlp.eval()
    # xent = nn.CrossEntropyLoss(reduction='sum')

    # Test set
    loss = 0.
    correct = 0.
    total = 0.
    with torch.no_grad():
        for batch_idx, ((raw_images, raw_labels), (aug1_images, aug1_labels), (aug2_images, aug2_labels)) in enumerate(val_train_loader):
            images, labels = [], []
            for i in range(raw_images.size(0)):
                images.append(raw_images[i])
                labels.append(raw_labels[i])
            images, labels = torch.stack(images).to(device), torch.stack(labels).to(device)

            _, pred = mlp(enc(images)).softmax(1).max(1)
            loss += xent(mlp(enc(images)).softmax(1), labels, reduction='sum').item()
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
    avg_loss = loss / total
    acc = 100. * correct / total

    args.logs['val_train_loss'][t] = avg_loss
    args.logs['val_train_acc'][t] = acc

    enc.train()
    mlp.train()
    return 0


def evaluation_test_sklearn(args, enc, classifier, test_imgs, test_labels, t):
    enc.eval()
    # xent = nn.CrossEntropyLoss(reduction='sum')

    # Test set
    x = enc(test_imgs).cpu().detach().numpy()
    distr = torch.from_numpy(classifier.predict_proba(x)).to(device)
    pred = torch.from_numpy(classifier.predict(x)).to(device)

    total = test_labels.size(0)
    avg_loss = xent(distr, test_labels, reduction='sum').item() / total
    acc = 100. * pred.eq(test_labels).sum().item() / total

    args.logs['test_loss'][t] = avg_loss
    args.logs['test_acc'][t] = acc

    enc.train()
    return 0


def pretrain(args, enc, val_train_loader):
    # Linear layer training
    enc.train()
    pretrain_opt = torch.optim.Adam(enc.parameters(), lr=args.pretrain_lr, weight_decay=args.decay)
    mlp = ClassifierModel(args.L, args.C).to(device).train()
    mlp_opt = torch.optim.Adam(mlp.parameters(), lr=args.mlp_lr, weight_decay=args.mlp_decay)
    # xent = nn.CrossEntropyLoss()

    for _ in range(args.nb_pretrain_epochs):
        for batch_idx, ((raw_images, raw_labels), (aug1_images, aug1_labels), (aug2_images, aug2_labels)) in enumerate(val_train_loader):
            assert raw_labels.tolist() == aug1_labels.tolist()
            assert raw_labels.tolist() == aug2_labels.tolist()
            pretrain_opt.zero_grad()
            mlp_opt.zero_grad()

            images, labels = [], []
            for i in range(raw_images.size(0)):
                images.append(raw_images[i])
                labels.append(raw_labels[i])
                # images.append(aug_images[i])  # pretrain with augmented images?
                # labels.append(aug_labels[i])
            images, labels = torch.stack(images).to(device), torch.stack(labels).to(device)

            distr = mlp(enc(images)).softmax(1)
            mlp_loss = xent(distr, labels)
            mlp_loss.backward()
            pretrain_opt.step()
            mlp_opt.step()

    return enc

def fine_tuning(args, enc, mlp, val_train_loader, val_test_loader, t):
    # Linear layer training
    mlp_opt = torch.optim.Adam(mlp.parameters(), lr=args.mlp_lr, weight_decay=args.mlp_decay)
    tuning_opt = torch.optim.Adam(enc.parameters(), lr=args.lr / 10, weight_decay=args.decay)
    # xent = nn.CrossEntropyLoss(reduction='sum')

    best_loss_test = 1e20
    for _ in range(args.E):
        enc.train()
        mlp.train()
        for batch_idx, ((raw_images, raw_labels), (aug1_images, aug1_labels), (aug2_images, aug2_labels)) in enumerate(val_train_loader):
            tuning_opt.zero_grad()
            mlp_opt.zero_grad()

            images, labels = [], []
            for i in range(raw_images.size(0)):
                images.append(raw_images[i])
                labels.append(raw_labels[i])
            images, labels = torch.stack(images).to(device), torch.stack(labels).to(device)

            distr = mlp(enc(images)).softmax(1)

            mlp_loss = xent(distr, labels, reduction='sum') / images.size(0)

            mlp_loss.backward()
            tuning_opt.step()
            mlp_opt.step()

        enc.eval()
        mlp.eval()
        with torch.no_grad():
            loss_test = 0.
            correct_test = 0.
            total_test = 0.
            for batch_idx, ((raw_images, raw_labels), (aug1_images, aug1_labels), (aug2_images, aug2_labels)) in enumerate(val_test_loader):
                images, labels = [], []
                for i in range(raw_images.size(0)):
                    images.append(raw_images[i])
                    labels.append(raw_labels[i])
                images, labels = torch.stack(images).to(device), torch.stack(labels).to(device)

                distr = mlp(enc(images)).softmax(1)
                _, pred = distr.max(1)

                loss_test += xent(distr, labels, reduction='sum').item()
                correct_test += pred.eq(labels).sum().item()
                total_test += images.size(0)
            if best_loss_test > loss_test / total_test:
                best_loss_test = loss_test / total_test
                evaluation_valtrain_linlayer(args, enc, mlp, val_train_loader, t)

                args.logs['val_test_loss'][t] = loss_test / total_test
                args.logs['val_test_acc'][t] = 100. * correct_test / total_test

                best_enc = deepcopy(enc.state_dict())
                best_mlp = deepcopy(mlp.state_dict())

                if args.best_loss > loss_test / total_test:
                    args.best_loss = loss_test / total_test
                    args.best_timestep = t

    enc.load_state_dict(best_enc)
    mlp.load_state_dict(best_mlp)

    enc.train()
    mlp.train()
    return enc, mlp