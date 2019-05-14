# from skimage.draw import random_shapes
from experiments.RandomShapes.random_shapes import random_shapes
import cv2
import numpy as np
import os


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

    def generate(self):
        img, annotations, indices = random_shapes(self.imSize, min_shapes=10, max_shapes=20, min_size=40, max_size=70)
        labelImg = np.zeros(img.shape[:2], np.uint8)
        # bbox is ((y1, y2), (x1, x2))
        for [className, bbox], ind in zip(annotations, indices):
            labelImg[ind] = self.labelValues[className]
        return img, labelImg

    def generateN(self, n):
        return (self.generate() for _ in range(n))

    def generateDatasetItem(self, n, datasetDir, imageDir, labelsDir):
        imageDir = os.path.join(datasetDir, imageDir)
        labelsDir = os.path.join(datasetDir, labelsDir)
        for i, (img, labels) in enumerate(self.generateN(n)):
            file = f'{i + 1}.png'
            cv2.imwrite(os.path.join(imageDir, file), Utils_.rgb2bgr(img))
            cv2.imwrite(os.path.join(labelsDir, file), img)


def main():
    generator = Generator([768, 1024])

    generator.generateDatasetItem(1000, 'dataset', 'train', 'trainannot')
    generator.generateDatasetItem(200, 'dataset', 'val', 'valannot')
    generator.generateDatasetItem(300, 'dataset', 'test', 'testannot')

    # Utils_.imshow(img, 'img')
    # Utils_.imshow(labelImg, 'label')
    # cv2.waitKeyEx()


if __name__ == '__main__':
    main()
