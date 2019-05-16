from experiments.random_shapes import random_shapes
from experiments.RndShapesDataset import RndShapesDataset
import cv2
import numpy as np
import os
from collections import OrderedDict


class Utils_:
    @staticmethod
    def rgb2bgr(rgb):
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @classmethod
    def imshow(cls, rgb, winname=''):
        cv2.imshow(winname, cls.rgb2bgr(rgb))


class Generator:
    labelValues = dict(circle=1, rectangle=2, triangle=3)

    def __init__(self, imSize):
        self.imSize = imSize

    def __fastAnnotation(self, labelImg, shapeBbox, labelValue):
        (y1, y2), (x1, x2) = shapeBbox
        centerX, centerY = [(x1 + x2) // 2, (y1 + y2) // 2]
        pt1 = (centerX - 5, centerY - 5)
        pt2 = (centerX + 5, centerY + 5)
        cv2.rectangle(labelImg, pt1, pt2, labelValue, -1)

    def generate(self):
        _, annotations, indices, colors = random_shapes(self.imSize, min_shapes=10, max_shapes=20, min_size=50,
                                                        max_size=80)
        img = np.full([*self.imSize, 3], 255, np.uint8)
        labelImg = np.zeros(self.imSize, np.uint8)
        # bbox is ((y1, y2), (x1, x2))
        for [className, bbox], ind, color in zip(annotations, indices, colors):
            if className == 'triangle':
                continue
            img[ind] = color
            self.__fastAnnotation(labelImg, bbox, self.labelValues[className])
        return img, labelImg

    def generateN(self, n):
        return (self.generate() for _ in range(n))

    def generateDatasetItem(self, n, datasetDir, imageDir, labelsDir):
        imageDir = os.path.join(datasetDir, imageDir)
        labelsDir = os.path.join(datasetDir, labelsDir)
        for i, (img, labels) in enumerate(self.generateN(n)):
            file = f'{i + 1}.png'
            cv2.imwrite(os.path.join(imageDir, file), Utils_.rgb2bgr(img))
            cv2.imwrite(os.path.join(labelsDir, file), labels)


def generateClassWeights(dataset_dir, height, width, batch_size, workers=4):
    import torchvision.transforms as transforms
    import transforms as ext_transforms
    from PIL import Image
    import torch.utils.data as data
    from data.utils import enet_weighing, median_freq_balancing

    image_transform = transforms.Compose(
        [transforms.Resize((height, width)),
         transforms.ToTensor()])

    label_transform = transforms.Compose([
        transforms.Resize((height, width), Image.NEAREST),
        ext_transforms.PILToLongTensor()
    ])
    train_set = RndShapesDataset(
        dataset_dir,
        transform=image_transform,
        label_transform=label_transform)
    train_loader = data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers)

    class_encoding = OrderedDict(train_set.color_encoding)
    del class_encoding['triangle'] # without triangle

    # Get number of classes to predict
    num_classes = len(class_encoding)

    class_weights = enet_weighing(train_loader, num_classes)
    np.save(os.path.join(dataset_dir, 'class_weights_ENet'), class_weights)

    class_weights = median_freq_balancing(train_loader, num_classes)
    np.save(os.path.join(dataset_dir, 'class_weights_mfb'), class_weights)


def main():
    generator = Generator([768, 1024])

    generator.generateDatasetItem(1000, 'FastAnnotatedRndShapes', 'train', 'trainannot')
    generator.generateDatasetItem(200, 'FastAnnotatedRndShapes', 'val', 'valannot')
    generator.generateDatasetItem(300, 'FastAnnotatedRndShapes', 'test', 'testannot')

    h, w = generator.imSize
    generateClassWeights('./FastAnnotatedRndShapes', h, w, 6)


if __name__ == '__main__':
    main()
