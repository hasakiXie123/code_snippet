import torch

## turn a label vector into a one_hot label vector
a = torch.tensor([0, 1, 0, 1]).resize(4, 1)
b = torch.nn.functional.one_hot(a).squeeze(dim=1)# [4, 1, 2] into [4, 2]
