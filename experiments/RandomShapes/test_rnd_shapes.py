import numpy as np
from models.enet import ENet
from experiments.RandomShapes.RndShapesDataset import RndShapesDataset
import torch
import os
from data import utils
from PIL import Image
import matplotlib.pyplot as plt
from experiments.RandomShapes.RndShapesDataset import color_encoding
from experiments.LongTensorToCHWRGBTensor import LongTensorToCHWRGBTensor
import torchvision.transforms as transforms


def loadImageTensor(item='test'):
    numOfFiles = len(os.listdir(f'./RndShapes/{item}/'))
    num = np.random.randint(1, numOfFiles + 1)
    image = Image.open(f'./RndShapes/{item}/{num}.png')
    gtLabel = Image.open(f'./RndShapes/{item}annot/{num}.png')

    image_transform = transforms.Compose(
        [transforms.Resize((768, 1024)),
         transforms.ToTensor()])

    tensor = image_transform(image)
    return tensor, image, gtLabel


def main_():
    tensor, image = loadImageTensor()

    # plt.axis('off')
    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.axis('off')
    ax2.axis('off')

    ax1.imshow(image)
    ax2.imshow(image)
    plt.show()


def main():
    classesNum = len(RndShapesDataset.color_encoding)
    model = ENet(classesNum)

    state = torch.load('./checkpoints/ENet', map_location='cpu')
    model.load_state_dict(state['state_dict'])
    # load image, resize and convert to mini-batch tensor

    tensor, image, gtLabel = loadImageTensor('train')
    inputMiniBatch = tensor.unsqueeze(0)

    gtLabel = torch.LongTensor(np.array(gtLabel))


    with torch.no_grad():
        predictions = model(inputMiniBatch)

    _, predictions = torch.max(predictions.data, 1)
    prediction = torch.unbind(predictions)[0]


    toRgb = LongTensorToCHWRGBTensor(color_encoding)
    prediction = toRgb(prediction).cpu().numpy().transpose(1, 2, 0)
    gtLabel = toRgb(gtLabel).cpu().numpy().transpose(1, 2, 0)

    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')

    ax1.imshow(image)
    ax2.imshow(prediction)
    ax3.imshow(gtLabel)
    plt.show()


main()
