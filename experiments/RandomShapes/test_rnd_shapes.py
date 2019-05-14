import numpy as np
from models.enet import ENet
from experiments.RandomShapes.RndShapesDataset import RndShapesDataset
import torch


def main():
    classesNum = len(RndShapesDataset.color_encoding)
    model = ENet(classesNum)

    state = torch.load('./checkpoints/ENet_', map_location='cpu')
    model.load_state_dict(state['state_dict'])
    # load image, resize and convert to mini-batch tensor

    inputMiniBatch = torch.tensor(np.zeros([1, 3, 768, 1024], np.float32))
    result = model(inputMiniBatch)
    print(result.dtype, result.shape)
    result = result.detach().numpy()
    result = result.argmax(axis=0)
    print(result.min(), result.max())


main()
