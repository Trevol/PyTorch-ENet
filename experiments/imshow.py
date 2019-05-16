import matplotlib.pyplot as plt


def to2dList(*arys):
    # list or npImage or pilImage
    result = []
    rows, cols = 0, 0
    for ar in arys:
        if not isinstance(ar, list):  # single item
            ar = [ar]
        elif len(ar) == 0:
            continue
        result.append(ar)
        rows += 1
        cols = max(cols, len(ar))
    return result, rows, cols


def imshow(*imagesGrid):
    imagesGrid, rows, cols = to2dList(*imagesGrid)

    f = plt.figure()
    for rowIndex, imRow in enumerate(imagesGrid):
        axPos = rowIndex * cols + 1
        for im in imRow:
            ax = f.add_subplot(rows, cols, axPos)
            ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                           labelbottom=False, labelleft=False)
            ax.imshow(im)
            axPos += 1

    plt.show()


if __name__ == '__main__':
    import numpy as np
    from PIL import Image


    def images(color):
        npImage = np.zeros([100, 100, 3], np.uint8)
        npImage = np.add(npImage, color, out=npImage, casting='unsafe')

        pilImage = Image.fromarray(npImage, 'RGB')
        return npImage, pilImage


    def main():
        npImageRed, pilImageRed = images([255, 0, 0])
        npImageGreen, pilImageGreen = images([0, 255, 0])
        imshow(npImageRed, [npImageGreen, pilImageGreen], pilImageRed)


    main()
