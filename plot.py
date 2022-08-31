import argparse
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

args_form = argparse.ArgumentParser(allow_abbrev=False)
args_form.add_argument("--s", type=int, default=0)
args_form.add_argument("--e", type=int, default=-1)
args_form.add_argument("--data", type=str, default="MNIST")
args_form.add_argument("--val_pc", type=int, default=0.05)
args_form.add_argument("--individual_plotting", action='store_true', default=False)
args_form.add_argument("--skip_linear", action='store_true', default=False)
args_form.add_argument("--train_type", type=str, default="finetune")

val_pc = [0.05, 0.025, 0.015, 0.01, 0.005]
seeds = [0,1,2,3,4]

args_orig = args_form.parse_args()
if args_orig.e == -1: args_orig.e = args_orig.s

nb_graphs = 7
fig, axarr = plt.subplots(nb_graphs, figsize=(5, nb_graphs * 3))
fname = [["./%s/logs_%s_%s_%s.pytorch" % (args_orig.train_type, args_orig.train_type, str(val), str(seed)) for val in val_pc] for seed in seeds]
# fname = [["./hyper/logs_%s_%s.pytorch" % (str(val), str(seed)) for val in val_pc] for seed in seeds]
for k in range(len(val_pc)):
    for s in range(len(seeds)):
        logs = torch.load(fname[s][k], map_location=device)
        args = logs["args"]

        metrics = list(args.logs.keys())
        if s == args_orig.s:
            titles = metrics
            xs = [[[] for _ in range(len(metrics))] for _ in range(len(val_pc))]
            ys = [[[] for _ in range(len(metrics))] for _ in range(len(val_pc))]

        for i, metric in enumerate(metrics):
            xs[k][i].append(list(args.logs[metric].keys()))
            ys[k][i].append(list(args.logs[metric].values()))


    for i in range(len(xs[k])):
        # if 'val_train_loss' in titles[i]: continue
        # if 'val_test_loss' in titles[i]: continue

        x = np.array(xs[k][i][0])
        y = np.array(ys[k][i])

        y_mean = y.mean(axis=0)
        y_std = y.std(axis=0)

        axarr[i].set_title(titles[i])
        axarr[i].plot(x, y_mean, label=str(int(val_pc[k] * 60000)))
        axarr[i].fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.5)
        axarr[i].legend()

savepath = "./%s/plots_summary_%s_%s.png" % (args_orig.train_type, args_orig.data, args_orig.train_type)

plt.tight_layout()
fig.savefig(savepath)
plt.close("all")
