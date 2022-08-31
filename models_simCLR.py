import torch
from torch import nn
import torch.nn.functional as F


def get_enc_sz(in_dim):
  if isinstance(in_dim, int):
    enc_sz = 32
  else:
    enc_sz = 256

  return enc_sz


def get_enc(in_dim, L):
  enc_sz = get_enc_sz(in_dim)
  if isinstance(in_dim, int):
    return nn.Sequential(
        nn.Linear(in_dim, enc_sz),
        nn.ReLU(),
        nn.Linear(enc_sz, L))

  elif in_dim[0] == 1:
    return nn.Sequential(
        nn.Conv2d(in_dim[0], 32, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # 14, 14
        nn.Flatten(start_dim=1),
        nn.Linear(64 * 14 * 14, enc_sz),
        nn.ReLU(),
        nn.Linear(enc_sz, L))

  elif in_dim[0] == 3:
    return nn.Sequential(
        nn.Conv2d(in_dim[0], 32, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # 16, 16
        nn.Flatten(start_dim=1),
        nn.Linear(64 * 16 * 16, enc_sz),
        nn.ReLU(),
        nn.Linear(enc_sz, L))

  else:
      print('Image dimensions error:', in_dim)
      raise NotImplementedError

class TargetModel(nn.Module):
  def __init__(self, in_dim, L):
    super().__init__()

    self.L = L
    self.feat_space = get_enc(in_dim, L)


  def forward(self, d_x):
    encoding = self.feat_space(d_x)
    encoding = F.normalize(encoding)

    return encoding



def trim(probs):
  new_probs = probs.clone()
  new_probs[probs < 1e-9] = 1e-9
  new_probs[probs > 1e9] = 1e9
  return new_probs


class ClassifierModel(nn.Module):
  def __init__(self, L, C):
    super().__init__()
    self.pred = nn.Sequential(
      nn.Linear(L, C),  # not softmaxed
    )

  def forward(self, d_x):
    return self.pred(d_x)

