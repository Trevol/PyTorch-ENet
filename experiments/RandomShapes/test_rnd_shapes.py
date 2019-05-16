import numpy as np
from models.enet import ENet
from experiments.RndShapesDataset import RndShapesDataset
import torch
import os
from PIL import Image
from experiments.RndShapesDataset import color_encoding
from experiments.LongTensorToCHWRGBTensor import LongTensorToCHWRGBTensor
import torchvision.transforms as transforms
from experiments.imshow import imshow
from experiments.RandomShapes.generate_dataset import Generator


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

    imageAsTensor, image, gtTestLabel = loadImageTensor('train')
    generatedImage, gtGeneratedLabel = Generator([768, 1024]).generate()

    inputMiniBatch = torch.stack([imageAsTensor, transforms.ToTensor()(generatedImage)])
    images = [image, generatedImage]
    gtLabels = [gtTestLabel, gtGeneratedLabel]

    # gtTestLabel = torch.LongTensor(np.array(gtTestLabel))

    with torch.no_grad():
        predictions = model(inputMiniBatch)

    _, predictions = torch.max(predictions.data, 1)
    toRgb = LongTensorToCHWRGBTensor(color_encoding)

    imageRows = []
    for prediction, image, gtLabel in zip(torch.unbind(predictions), images, gtLabels):
        predictedLabels = toRgb(prediction).cpu().numpy().transpose(1, 2, 0)
        # gtLabelImage = toRgb(gtTestLabel).cpu().numpy().transpose(1, 2, 0)

        imageRows.append(
            [image, predictedLabels, gtLabel]
        )

    imshow(*imageRows)


main()
