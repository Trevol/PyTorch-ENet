import numpy as np
from models.enet import ENet
from experiments.RndShapesDataset import color_encoding as class_encoding
import torch
import os
from PIL import Image
from experiments.LongTensorToCHWRGBTensor import LongTensorToCHWRGBTensor
import torchvision.transforms as transforms
from experiments.imshow import imshow
from experiments.FastAnnotatedRandomShapes.generate import Generator
from collections import OrderedDict

def loadImageTensor(item='test'):
    numOfFiles = len(os.listdir(f'./FastAnnotatedRndShapes/{item}/'))
    num = np.random.randint(1, numOfFiles + 1)
    image = Image.open(f'./FastAnnotatedRndShapes/{item}/{num}.png')
    gtLabel = Image.open(f'./FastAnnotatedRndShapes/{item}annot/{num}.png')

    image_transform = transforms.Compose(
        [transforms.Resize((768, 1024)),
         transforms.ToTensor()])

    tensor = image_transform(image)
    return tensor, image, gtLabel


def main():
    color_encoding = OrderedDict(class_encoding)
    del color_encoding['triangle']

    classesNum = len(color_encoding)
    model = ENet(classesNum)

    state = torch.load('./checkpoints/ENet_650', map_location='cpu')
    model.load_state_dict(state['state_dict'])

    imageAsTensor, image, gtTestLabel = loadImageTensor('train')
    generatedImage, gtGeneratedLabel = Generator([768, 1024]).generate()
    blankImage = np.full([768, 1024, 3], 255, np.uint8)
    gtBlankLabel = np.full([768, 1024], 0, np.uint8)

    inputMiniBatch = torch.stack([
        imageAsTensor,
        transforms.ToTensor()(generatedImage),
        transforms.ToTensor()(blankImage)
    ])
    images = [image, generatedImage, blankImage]
    gtLabels = [gtTestLabel, gtGeneratedLabel, gtBlankLabel]

    with torch.no_grad():
        predictions = model(inputMiniBatch)

    _, predictions = torch.max(predictions.data, 1)
    toRgb = LongTensorToCHWRGBTensor(color_encoding)

    imageRows = []
    for prediction, image, gtLabel in zip(torch.unbind(predictions), images, gtLabels):
        predictedLabels = toRgb(prediction).cpu().numpy().transpose(1, 2, 0)

        imageRows.append(
            [image, predictedLabels, gtLabel]
        )

    imshow(*imageRows)


main()
