import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
from collections import OrderedDict, defaultdict
import sys
from datetime import datetime
import argparse
from generic import *
import shutil
import os
from data_simCLR import *
from models_simCLR import *
from util_repr import *
from copy import deepcopy
import matplotlib.pyplot as plt

args_form = argparse.ArgumentParser(allow_abbrev=False)
args_form.add_argument("--train_type", type=str, default="basic")  # basic, pretrain, finetune
args_form.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4])

args_form.add_argument("--data", type=str, default="MNIST")
# args_form.add_argument("--best_acc", type=float, default=0)
args_form.add_argument("--best_loss", type=float, default=1e20)
args_form.add_argument("--best_timestep", type=int, default=0)

args_form.add_argument("--max_epoch", type=int, default=100)  # number of opt steps/epochs
args_form.add_argument("--J", type=int, default=10)  # nb training steps of encoder before training a lin layer on top
args_form.add_argument("--K", type=int, default=50)  # number of steps to train the lin. layer on top
args_form.add_argument("--L", type=int, default=256)  # repr dim
args_form.add_argument("--batch_sz", type=int, default=128)

args_form.add_argument("--lr", type=float, default=1e-2)
args_form.add_argument("--decay", type=float, default=1e-5)

args_form.add_argument("--mlp_lr", type=float, default=1e-2)
args_form.add_argument("--mlp_decay", type=float, default=1e-5)

args_form.add_argument("--pretrain_lr", type=float, default=1e-2)
args_form.add_argument("--nb_pretrain_epochs", type=int, default=5)
args_form.add_argument("--E", type=int, default=30)

args_form.add_argument("--val_pc", type=float, default=0.05)  # .05, .025, 0.015, 0.01, 0.005, 0.005
args_form.add_argument("--tau", type=float, default=0.1)  # 0.05, 0.1, 0.5, 1.0

args_form.add_argument("--visualize_data", default=False, action="store_true")
args_form.add_argument("--random_hyper", default=False, action="store_true")
args = args_form.parse_args()

if args.random_hyper: select_random_hyperparams(args)
if args.train_type == 'finetune_no_pretrain': finetune_no_pt = True
else: finetune_no_pt = False


for seed in args.seeds:

    if finetune_no_pt: args.train_type = 'finetune_no_pretrain'
    set_seed(seed)
    print(datetime.now())
    args.logs = defaultdict(OrderedDict)
    print(args)
    name = "./%s/logs_%s_%s_%s.pytorch" % (str(args.train_type), str(args.train_type), str(args.val_pc), str(seed))
    # name = "./hyper/logs_%s_%s.pytorch" % (str(args.tau), str(args.lr))
    args.best_loss = 1e20  # Reinitialization

    # Data
    train_imgs, \
    val_train_imgs, val_train_labels, val_train_loader, \
    val_test_imgs, val_test_labels, val_test_loader, \
    test_imgs, test_labels, test_loader = get_data(args)

    # Models and initial state
    enc = TargetModel(args.in_dim, args.L).to(device).train()
    enc_opt = torch.optim.Adam(enc.parameters(), lr=args.lr, weight_decay=args.decay)  # , momentum=0.9
    head = ClassifierModel(args.L, args.L).to(device).train()
    head_opt = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=args.decay)

    cos_dist = nn.CosineSimilarity()

    if args.train_type == 'persistent_finetune':
        mlp = ClassifierModel(args.L, args.C).to(device).train()
    elif args.train_type == 'pretrain' or args.train_type == 'finetune':
        enc = pretrain(args, enc, val_train_loader)
    elif args.train_type == 'finetune_no_pretrain':
        args.train_type = 'finetune'  # skip pre-training part, but act the same
    elif not args.train_type == 'basic':
        print('Please choose --train_type ("basic", "pretrain", "finetune", "persistent_finetune", "finetune_no_pretrain")')
        raise NotImplementedError

    t = 0
    for e in range(args.max_epoch):

        torch.save({"args": args}, name)
        sys.stdout.flush()

        if args.train_type == "finetune" or args.train_type == "persistent_finetune":
            if args.train_type == "finetune": mlp = ClassifierModel(args.L, args.C).to(device).train()
            enc, mlp = fine_tuning(args, enc, mlp, val_train_loader, val_test_loader, t)
            enc_opt = torch.optim.Adam(enc.parameters(), lr=args.lr, weight_decay=args.decay)
            evaluation_test_linlayer(args, enc, mlp, test_loader, t)
        else:
            classifier = logistic_reg(args, enc, val_train_imgs[:, 0], val_train_labels[:, 0], val_test_imgs, val_test_labels, t)
            evaluation_test_sklearn(args, enc, classifier, test_imgs, test_labels, t)

        # Encoder training
        enc.train()
        train_total = 0.
        for j in range(args.J):
            enc_opt.zero_grad()
            head_opt.zero_grad()

            d_j = np.random.choice(len(train_imgs), size=args.batch_sz >> 1, replace=False)

            a = head(enc(train_imgs[d_j, 1]))
            a_norm = torch.norm(a, dim=1).reshape(-1, 1).to(device)
            a_cap = torch.div(a, a_norm).to(device)

            b = head(enc(train_imgs[d_j, 2]))
            b_norm = torch.norm(b, dim=1).reshape(-1, 1).to(device)
            b_cap = torch.div(b, b_norm).to(device)

            a_cap_b_cap = torch.cat([a_cap, b_cap], dim=0).to(device)
            b_cap_a_cap = torch.cat([b_cap, a_cap], dim=0).to(device)

            sim = torch.mm(a_cap_b_cap, a_cap_b_cap.t())
            exp_sim_by_tau = torch.exp(torch.div(sim, args.tau))
            mask = torch.ones_like(exp_sim_by_tau, device=device) - torch.eye(exp_sim_by_tau.shape[0], exp_sim_by_tau.shape[1], device=device)
            exp_sim_by_tau = mask * exp_sim_by_tau

            numerators = torch.exp(torch.div(cos_dist(a_cap_b_cap, b_cap_a_cap), args.tau))
            denominators = torch.sum(exp_sim_by_tau, dim=1)
            num_by_den = torch.div(numerators, denominators)
            neglog_num_by_den = -torch.log(num_by_den)
            train_loss = neglog_num_by_den.mean()

            train_loss.backward()
            enc_opt.step()
            head_opt.step()

            args.logs['train_loss'][t] = train_loss.item()

            t += 1  # nb of opt. steps taken
            # t += args.batch_sz  # nb of images seen


        print("-------------- seed: %d --- t: %d --------------" % (seed, t-args.J))
        print(datetime.now())
        print("val_pc: %s  ; lr: %s  ; wd: %s" % (args.val_pc, args.lr, args.decay))
        print("Train     Loss : %.4f" % args.logs['train_loss'][t - args.J])
        print("Test      Loss : %.9f" % args.logs['test_loss'][t - args.J])
        print("Val. Test Loss : %.9f" % args.logs['val_test_loss'][t-args.J])
        print("Val. Train Loss: %.9f" % args.logs['val_train_loss'][t-args.J])
        print("Test       Acc : %.2f" % args.logs['test_acc'][t - args.J])
        print("Val. Test  Acc : %.2f" % args.logs['val_test_acc'][t-args.J])
        print("Val. Train Acc : %.2f \n" % args.logs['val_train_acc'][t-args.J])
        print('Best Loss: %.6f' % args.best_loss)
        print('Best Time: %.d' % args.best_timestep)

    torch.save({"args": args}, name)
