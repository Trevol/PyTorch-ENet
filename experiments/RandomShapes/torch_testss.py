import numpy as np
import torch


def main():
    a1 = np.array([1, 1])
    a2 = np.array([2, 2])
    t1 = torch.from_numpy(a1)
    t2 = torch.from_numpy(a2)
    # torch.from_numpy()
    stack = torch.stack([t1, t2])
    print(stack)



main()
