from skimage.draw import random_shapes
import cv2


def main():
    def rgb2bgr(rgb):
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def imshow(rgb, winname=''):
        cv2.imshow(winname, rgb2bgr(rgb))

    img, annotations = random_shapes([768, 1024], min_shapes=10, max_shapes=40)
    for className, ((y1, y2), (x1, x2)) in annotations:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0))
    imshow(img)

    cv2.waitKeyEx()


if __name__ == '__main__':
    main()
