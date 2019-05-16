import numpy as np
from models.enet import ENet
from experiments.RandomShapes.RndShapesDataset import RndShapesDataset
import torch
import os
from PIL import Image
from experiments.RandomShapes.RndShapesDataset import color_encoding
from experiments.LongTensorToCHWRGBTensor import LongTensorToCHWRGBTensor
import torchvision.transforms as transforms
from experiments.imshow import imshow
from experiments.random_shapes import random_shapes


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


def main():
    classesNum = len(RndShapesDataset.color_encoding)
    model = ENet(classesNum)

    state = torch.load('./checkpoints/ENet', map_location='cpu')
    model.load_state_dict(state['state_dict'])
    # load image, resize and convert to mini-batch tensor

    tensor, image, gtLabel = loadImageTensor('train')
    # inputMiniBatch = tensor.unsqueeze(0)
    inputMiniBatch = torch.stack([tensor, tensor])

    gtLabel = torch.LongTensor(np.array(gtLabel))

    with torch.no_grad():
        predictions = model(inputMiniBatch)

    _, predictions = torch.max(predictions.data, 1)
    toRgb = LongTensorToCHWRGBTensor(color_encoding)

    imageRows = []
    for prediction in torch.unbind(predictions):
        predictedLabels = toRgb(prediction).cpu().numpy().transpose(1, 2, 0)
        gtLabelImage = toRgb(gtLabel).cpu().numpy().transpose(1, 2, 0)

        imageRows.append(
            [image, predictedLabels, gtLabelImage]
        )

    imshow(*imageRows)


main()
