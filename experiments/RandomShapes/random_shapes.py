import numpy as np
from skimage.draw._random_shapes import _generate_random_colors, SHAPE_CHOICES, SHAPE_GENERATORS
from skimage._shared.utils import warn


def random_shapes(image_shape,
                  max_shapes,
                  min_shapes=1,
                  min_size=2,
                  max_size=None,
                  multichannel=True,
                  num_channels=3,
                  shape=None,
                  intensity_range=None,
                  allow_overlap=False,
                  num_trials=100,
                  random_seed=None):
    """Generate an image with random shapes, labeled with bounding boxes.

    The image is populated with random shapes with random sizes, random
    locations, and random colors, with or without overlap.

    Shapes have random (row, col) starting coordinates and random sizes bounded
    by `min_size` and `max_size`. It can occur that a randomly generated shape
    will not fit the image at all. In that case, the algorithm will try again
    with new starting coordinates a certain number of times. However, it also
    means that some shapes may be skipped altogether. In that case, this
    function will generate fewer shapes than requested.

    Parameters
    ----------
    image_shape : tuple
        The number of rows and columns of the image to generate.
    max_shapes : int
        The maximum number of shapes to (attempt to) fit into the shape.
    min_shapes : int, optional
        The minimum number of shapes to (attempt to) fit into the shape.
    min_size : int, optional
        The minimum dimension of each shape to fit into the image.
    max_size : int, optional
        The maximum dimension of each shape to fit into the image.
    multichannel : bool, optional
        If True, the generated image has ``num_channels`` color channels,
        otherwise generates grayscale image.
    num_channels : int, optional
        Number of channels in the generated image. If 1, generate monochrome
        images, else color images with multiple channels. Ignored if
        ``multichannel`` is set to False.
    shape : {rectangle, circle, triangle, None} str, optional
        The name of the shape to generate or `None` to pick random ones.
    intensity_range : {tuple of tuples of uint8, tuple of uint8}, optional
        The range of values to sample pixel values from. For grayscale images
        the format is (min, max). For multichannel - ((min, max),) if the
        ranges are equal across the channels, and ((min_0, max_0), ... (min_N, max_N))
        if they differ. As the function supports generation of uint8 arrays only,
        the maximum range is (0, 255). If None, set to (0, 254) for each
        channel reserving color of intensity = 255 for background.
    allow_overlap : bool, optional
        If `True`, allow shapes to overlap.
    num_trials : int, optional
        How often to attempt to fit a shape into the image before skipping it.
    seed : int, optional
        Seed to initialize the random number generator.
        If `None`, a random seed from the operating system is used.

    Returns
    -------
    image : uint8 array
        An image with the fitted shapes.
    labels : list
        A list of labels, one per shape in the image. Each label is a
        (category, ((r0, r1), (c0, c1))) tuple specifying the category and
        bounding box coordinates of the shape.

    Examples
    --------
    >>> import skimage.draw
    >>> image, labels = skimage.draw.random_shapes((32, 32), max_shapes=3)
    >>> image # doctest: +SKIP
    array([
       [[255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        ...,
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255]]], dtype=uint8)
    >>> labels # doctest: +SKIP
    [('circle', ((22, 18), (25, 21))),
     ('triangle', ((5, 6), (13, 13)))]
    """
    if min_size > image_shape[0] or min_size > image_shape[1]:
        raise ValueError('Minimum dimension must be less than ncols and nrows')
    max_size = max_size or max(image_shape[0], image_shape[1])

    if not multichannel:
        num_channels = 1

    if intensity_range is None:
        intensity_range = (0, 254) if num_channels == 1 else ((0, 254),)
    else:
        tmp = (intensity_range,) if num_channels == 1 else intensity_range
        for intensity_pair in tmp:
            for intensity in intensity_pair:
                if not (0 <= intensity <= 255):
                    msg = 'Intensity range must lie within (0, 255) interval'
                    raise ValueError(msg)

    random = np.random.RandomState(random_seed)
    user_shape = shape
    image_shape = (image_shape[0], image_shape[1], num_channels)
    image = np.full(image_shape, 255, dtype=np.uint8)
    filled = np.zeros(image_shape, dtype=bool)
    labels = []
    allIndices = []

    num_shapes = random.randint(min_shapes, max_shapes + 1)
    colors = _generate_random_colors(num_shapes, num_channels,
                                     intensity_range, random)
    for shape_idx in range(num_shapes):
        if user_shape is None:
            shape_generator = random.choice(SHAPE_CHOICES)
        else:
            shape_generator = SHAPE_GENERATORS[user_shape]
        shape = (min_size, max_size)
        for _ in range(num_trials):
            # Pick start coordinates.
            column = random.randint(image_shape[1])
            row = random.randint(image_shape[0])
            point = (row, column)
            try:
                indices, label = shape_generator(point, image_shape, shape,
                                                 random)
            except ArithmeticError:
                # Couldn't fit the shape, skip it.
                continue
            # Check if there is an overlap where the mask is nonzero.
            if allow_overlap or not filled[indices].any():
                image[indices] = colors[shape_idx]
                filled[indices] = True
                labels.append(label)
                allIndices.append(indices)
                break
        else:
            warn('Could not fit any shapes to image, '
                 'consider reducing the minimum dimension')

    if not multichannel:
        image = np.squeeze(image, axis=2)
    return image, labels, allIndices
