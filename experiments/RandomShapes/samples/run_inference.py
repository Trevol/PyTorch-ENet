import keras
import tensorflow as tf

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

from random_shapes import randomShapes, addSpacers

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def main():
    # use this environment flag to change which GPU to use
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # set the modified tf session as backend in keras
    keras.backend.tensorflow_backend.set_session(get_session())

    # cell 2
    # adjust this to point to your downloaded/trained model
    # models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
    # model_path = os.path.join('..', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')
    model_path = 'resnet50_random_shapes_07_ellipses.h5'
    # load retinanet model
    model = models.load_model(model_path, backbone_name='resnet50')

    # if the model is not converted to an inference model, use the line below
    # see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
    # model = models.convert_model(model)

    # print(model.summary())

    # load label to names mapping for visualization purposes
    labels_to_names = {0: 'circle', 1: 'ellipse', 2: 'rectangle', 3: 'triangle'}

    image, annotations = randomShapes((768, 1024))
    image = addSpacers(image, 200)
    predict_on_image(model, labels_to_names, image)

    image, annotations = randomShapes((768, 1024))
    image = addSpacers(image, 200)
    predict_on_image(model, labels_to_names, image)

    image, annotations = randomShapes((768, 1024))
    image = addSpacers(image, 200)
    # random noise
    image = np.uint8(image - 60 + np.random.randint(2, 55, image.shape))
    predict_on_image(model, labels_to_names, image)

    image = randomEllipses()
    image = addSpacers(image, 200)
    predict_on_image(model, labels_to_names, image)


def randomEllipses():
    img = np.full((768, 1024, 3), 255, np.uint8)
    cv2.ellipse(img, (500, 300), (30, 70), 0, 0, 360, (0, 255, 0), -1)
    return img


def predict_on_image(model, labels_to_names, image):
    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()


if __name__ == '__main__':
    main()
