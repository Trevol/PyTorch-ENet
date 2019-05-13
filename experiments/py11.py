import torch
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torchvision
from contextlib import contextmanager

# device = torch.device('cuda:0')
device = torch.device('cpu')


# tensor = torch.randn([3, 400, 600], dtype=torch.float32, device=device)

@contextmanager
def timeit():
    t0 = time.time()
    yield None
    t1 = time.time()
    print(f'{t1 - t0:.4f}')


def prepareTensor(inputImg):
    inputTensor = torch.tensor(inputImg)
    # print(inputTensor.shape, inputTensor.dtype)
    inputTensor = inputTensor.unsqueeze(0)
    # print(inputTensor.shape, inputTensor.dtype)
    # inputTensor = inputTensor.float()
    # print(inputTensor.shape, inputTensor.dtype)

    inputTensor = inputTensor.transpose(2, 3).transpose(1, 2)
    # print(inputTensor.shape, inputTensor.dtype)


def prepareTensor2(inputImg):
    inputTensor = torch.tensor(inputImg)
    # print(inputTensor.shape, inputTensor.dtype)

    inputTensor = inputTensor.transpose(1, 2).transpose(0, 1)
    # print(inputTensor.shape, inputTensor.dtype)

    inputTensor = inputTensor.unsqueeze(0)
    # print(inputTensor.shape, inputTensor.dtype)


def prepareTensorWithNumpy(inputImg: np.ndarray):
    inputImg = inputImg.transpose(2, 0, 1)
    # print(inputImg.shape, inputImg.dtype)

    inputImg = np.expand_dims(inputImg, 0)
    # print(inputImg.shape, inputImg.dtype)

    inputTensor = torch.tensor(inputImg)
    # print(inputTensor.shape, inputTensor.dtype)


def loadImage():
    imageFileName = 'Seq05VD_f05100.png'
    inputImg = plt.imread('./CamVid/test/' + imageFileName)
    return inputImg


def measure(inputImg, iters=1000):
    print('-------inputImg-----')

    print(inputImg.shape, inputImg.dtype)

    print('-------prepareTensor-----')
    with timeit():
        for _ in range(iters):
            prepareTensor(inputImg)

    print('-------prepareTensor2-----')
    with timeit():
        for _ in range(iters):
            prepareTensor2(inputImg)

    print('-------prepareTensorWithNumpy-----')
    with timeit():
        for _ in range(iters):
            prepareTensorWithNumpy(inputImg)


def main():
    inputImg = loadImage()
    # inputImg = cv2.resize(inputImg, (512, 512), cv2.INTER_NEAREST)
    measure(inputImg, 10000)
    measure(inputImg, 10000)


main()
