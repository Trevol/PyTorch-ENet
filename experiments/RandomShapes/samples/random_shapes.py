import numpy as np
import skimage.draw
from skimage.draw._random_shapes import SHAPE_CHOICES, SHAPE_GENERATORS


def randomShapes(imgShape):
    return skimage.draw.random_shapes(imgShape, max_shapes=40, min_size=30, max_size=60, intensity_range=((100, 255),))


def _generate_ellipse_mask(point, image, shape, random):
    """Generate a mask for a filled ellipse shape.

    The radiuses of the ellipse is generated randomly.

    Parameters
    ----------
    point : tuple
        The row and column of the top left corner of the rectangle.
    image : tuple
        The height, width and depth of the image into which the shape is placed.
    shape : tuple
        The minimum and maximum size and color of the shape to fit.
    random : np.random.RandomState
        The random state to use for random sampling.

    Raises
    ------
    ArithmeticError
        When a shape cannot be fit into the image with the given starting
        coordinates. This usually means the image dimensions are too small or
        shape dimensions too large.

    Returns
    -------
    label : tuple
        A (category, ((r0, r1), (c0, c1))) tuple specifying the category and
        bounding box coordinates of the shape.
    indices : 2-D array
        A mask of indices that the shape fills.
    """
    if shape[0] == 1 or shape[1] == 1:
        raise ValueError('size must be > 1 for ellipses')
    min_radius = shape[0] / 2.0
    max_radius = shape[1] / 2.0
    left = point[1]
    right = image[1] - point[1]
    top = point[0]
    bottom = image[0] - point[0]
    available_radius = min(left, right, top, bottom, max_radius)
    if available_radius < min_radius:
        raise ArithmeticError('cannot fit shape to image')

    r_radius = random.randint(min_radius, available_radius + 1)
    random_scale = random.randint(13, 26) / 10  # [1.3 ... 2.5]
    c_radius = np.int32(r_radius / random_scale)  # r_radius / [1.3 ... 2.5]

    r_radius, c_radius = random.permutation([r_radius, c_radius])

    # TODO: apply rotation (from -PI to PI)
    ellipse = skimage.draw.ellipse(point[0], point[1], r_radius, c_radius, rotation=0)

    label = ('ellipse', ((point[0] - r_radius + 1, point[0] + r_radius),
                         (point[1] - c_radius + 1, point[1] + c_radius)))

    return ellipse, label


SHAPE_GENERATORS['ellipse'] = _generate_ellipse_mask
SHAPE_CHOICES.append(_generate_ellipse_mask)


def addSpacers(img, color=255, size=40):
    h, w = img.shape[:2]
    d = None
    if len(img.shape) == 3:
        d = img.shape[-1]

    topSpacerShape = (size, w) if d is None else (size, w, d)
    topSpacer = np.empty(topSpacerShape, img.dtype)
    topSpacer[..., :] = color
    img = np.vstack([topSpacer, img])

    rigthSpacerShape = (h + size, size) if d is None else (h + size, size, d)
    rigthSpacer = np.empty(rigthSpacerShape, img.dtype)
    rigthSpacer[..., :] = color
    img = np.hstack([img, rigthSpacer])

    return img


if __name__ == '__main__':
    import cv2


    def main():
        imgShape = (768, 1024)
        img, annotations = randomShapes(imgShape)
        for className, ((y1, y2), (x1, x2)) in annotations:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0))

        img = addSpacers(img, color=255)
        cv2.imshow('', img)
        cv2.waitKey()


    def main_():
        img = np.zeros([1, 1, 3], np.uint8)
        img[..., :] = [3, 4, 5]
        print(img)

        img = np.zeros([2, 2], np.uint8)
        img[..., :] = 3
        print(img)


    main()
