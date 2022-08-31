import random
import numpy as np
import torch

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

def select_random_hyperparams(args):
  args.lr = random.choice((1e-1, 1e-2, 1e-3, 1e-4))
  args.decay = random.choice((1e-3, 1e-4, 1e-5))
  args.tau = random.choice((1., 0.5, 0.1, 0.05))

def xent(distr, labels, reduction='mean'):

  logs = - torch.sum(torch.log(torch.gather(distr, 1, labels.unsqueeze(1))))
  if reduction == 'mean':
    return logs / labels.size(0)
  if reduction == 'sum':
    return logs
