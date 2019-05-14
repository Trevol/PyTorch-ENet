import os
import torch
import torch.optim as optim
import torch.nn as nn

import torchvision.transforms as transforms
from PIL import Image
import transforms as ext_transforms
import torch.utils.data as data
import utils
from data.utils import enet_weighing, median_freq_balancing
from models.enet import ENet
from metric.iou import IoU
from test import Test
from collections import OrderedDict
import torchvision
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda')


class CityscapesArgs:
    from data import Cityscapes as ds
    dataset_ctor = ds
    dataset = 'Cityscapes'
    dataset_dir = './Cityscapes/'
    save_dir = '../save/ENet_Cityscapes/'
    height = 360
    width = 480
    batch_size = 2
    workers = 4
    mode = 'test'
    imshow_batch = False
    weighing = 'ENet_'
    ignore_unlabeled = True
    print_step = False
    name = 'ENet_'


class CamVidArgs:
    from data import CamVid as ds
    dataset_ctor = ds
    dataset = 'CamVid'
    dataset_dir = './CamVid/'
    save_dir = './save/'
    height = 360
    width = 480
    batch_size = 1
    workers = 4
    mode = 'test'
    imshow_batch = True
    weighing = 'ENet_'
    ignore_unlabeled = True
    print_step = False
    name = 'ENet_'


args = CamVidArgs


def loadTestDataset(datasetCtor):
    image_transform = transforms.Compose(
        [transforms.Resize((args.height, args.width)),
         transforms.ToTensor()])

    label_transform = transforms.Compose([
        transforms.Resize((args.height, args.width), Image.NEAREST),
        ext_transforms.PILToLongTensor()
    ])

    # Load the test set as tensors
    test_set = datasetCtor(
        args.dataset_dir,
        mode='test',
        transform=image_transform,
        label_transform=label_transform)
    test_loader = data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers)
    # Get encoding between pixel values in label images and RGB colors
    class_encoding = test_set.color_encoding

    # Remove the road_marking class from the CamVid dataset as it's merged
    # with the road class
    if args.dataset.lower() == 'camvid':
        del class_encoding['road_marking']

    return test_loader, class_encoding


def predict(model, images, class_encoding):
    images = images.to(device)

    # Make predictions!
    # model.eval()
    with torch.no_grad():
        predictions = model(images)

    # Predictions is one-hot encoded with "num_classes" channels.
    # Convert it to a single int using the indices where the maximum (1) occurs
    _, predictions = torch.max(predictions.data, 1)

    # label_to_rgb = transforms.Compose([
    #     ext_transforms.LongTensorToRGBPIL(class_encoding),
    #     transforms.ToTensor()
    # ])
    label_to_rgb = transforms.Compose([
        LongTensorToCHWRGBTensor(class_encoding)
    ])
    color_predictions = utils.batch_transform(predictions.cpu(), label_to_rgb)
    return color_predictions


class LongTensorToCHWRGBTensor(object):

    def __init__(self, rgb_encoding):
        self.rgb_encoding = rgb_encoding

    def __call__(self, tensor):
        """Performs the conversion from ``torch.LongTensor`` to a ``RGB tensor``

        Keyword arguments:
        - tensor (``torch.LongTensor``): the tensor to convert

        Returns:
        A ``RGB tensor of shape (C, H, W)``.

        """
        # Check if label_tensor is a LongTensor
        if not isinstance(tensor, torch.LongTensor):
            raise TypeError("label_tensor should be torch.LongTensor. Got {}"
                            .format(type(tensor)))
        # Check if encoding is a ordered dictionary
        if not isinstance(self.rgb_encoding, OrderedDict):
            raise TypeError("encoding should be an OrderedDict. Got {}".format(
                type(self.rgb_encoding)))

        # label_tensor might be an image without a channel dimension, in this
        # case unsqueeze it
        if len(tensor.size()) == 2:
            tensor.unsqueeze_(0)
        # (C, H, W) tensor
        color_tensor = torch.ByteTensor(3, tensor.size(1), tensor.size(2))

        for index, (class_name, color) in enumerate(self.rgb_encoding.items()):
            # Get a mask of elements equal to index
            mask = torch.eq(tensor, index).squeeze_()
            # Fill color_tensor with corresponding colors
            for channel, color_value in enumerate(color):
                color_tensor[channel].masked_fill_(mask, color_value)
        return color_tensor


def imshow_batch(images, labels):
    """Displays two grids of images. The top grid displays ``images``
    and the bottom grid ``labels``

    Keyword arguments:
    - images (``Tensor``): a 4D mini-batch tensor of shape
    (B, C, H, W)
    - labels (``Tensor``): a 4D mini-batch tensor of shape
    (B, C, H, W)

    """

    # Make a grid with the images and labels and convert it to numpy
    images = torchvision.utils.make_grid(images).numpy().transpose(1, 2, 0)
    labels = torchvision.utils.make_grid(labels).numpy().transpose(1, 2, 0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7))
    ax1.imshow(images)
    ax2.imshow(labels)

    plt.show()


def main():
    test_loader, class_encoding = loadTestDataset(args.dataset_ctor)

    num_classes = len(class_encoding)
    model = ENet(num_classes).to(device)

    modelFile = os.path.join(args.save_dir, args.name)
    checkpoint = torch.load(modelFile)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    images, _ = iter(test_loader).next()

    colorPredictions = predict(model, images, class_encoding)
    imshow_batch(images, colorPredictions)


main()
