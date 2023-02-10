import torch

## turn a label vector into a one_hot label vector
a = torch.tensor([0, 1, 0, 1]).resize(4, 1)
b = torch.nn.functional.one_hot(a).squeeze(dim=1)# [4, 1, 2] into [4, 2]

## replace the empty row of a vector with something else
zeros = torch.zeros([4, 2048])
one = torch.ones([1, 2048])
non_idxs = torch.all(zeros[..., :] == 0, axis=1).nonzero().tolist()
non_idxs = [x[0] for x in non_idxs]
zeros[non_idxs] = one
