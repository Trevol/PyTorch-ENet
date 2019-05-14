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
            cv2.imwrite(os.path.join(labelsDir, file), labels)


def generateClassWeights(dataset_dir, height, width, batch_size, workers=4):
    from experiments.RandomShapes.RndShapesDataset import RndShapesDataset as dataset
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
    train_set = dataset(
        dataset_dir,
        transform=image_transform,
        label_transform=label_transform)
    train_loader = data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers)

    class_encoding = train_set.color_encoding
    # Get number of classes to predict
    num_classes = len(class_encoding)

    class_weights = enet_weighing(train_loader, num_classes)
    np.save(os.path.join(dataset_dir, 'class_weights_ENet'), class_weights)

    class_weights = median_freq_balancing(train_loader, num_classes)
    np.save(os.path.join(dataset_dir, 'class_weights_mfb'), class_weights)

def main():
    generator = Generator([768, 1024])

    generator.generateDatasetItem(1000, 'RndShapes', 'train', 'trainannot')
    generator.generateDatasetItem(200, 'RndShapes', 'val', 'valannot')
    generator.generateDatasetItem(300, 'RndShapes', 'test', 'testannot')

    h, w = generator.imSize
    generateClassWeights('./RndShapes', h, w, 6)



if __name__ == '__main__':
    main()
