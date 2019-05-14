import numpy as np

import cv2
import os
import io
import sys
from keras_retinanet.bin.train import main as trainRetina
from random_shapes import randomShapes


def generateDataset(dir, filesNum):
    imgShape = (768, 1024)
    classNames = set()
    annotationsFileName = os.path.join(dir, 'annotations.csv')
    with open(annotationsFileName, 'w') as annotationsFile:
        emptyFile = 'empty.jpg'
        cv2.imwrite(os.path.join(dir, emptyFile), np.full([*imgShape, 3], 255, np.uint8))
        annotationsFile.write(f'{emptyFile},,,,,\n')

        for i in range(filesNum):
            img, annotations = randomShapes(imgShape)
            imageFile = f'{i + 1}.jpg'
            if len(annotations) == 0:
                annotationsFile.write(f'{imageFile},,,,,\n')
            for className, ((y1, y2), (x1, x2)) in annotations:
                annotationsFile.write(f'{imageFile},{x1},{y1},{x2},{y2},{className}\n')
                classNames.add(className)
            cv2.imwrite(os.path.join(dir, imageFile), img)
    return classNames, annotationsFileName


def saveClassMapping(dir, classNames):
    classNames = sorted(classNames)
    classMappingFileName = os.path.join(dir, 'class_mapping.csv')
    with open(classMappingFileName, 'w') as classMappingFile:
        for id, className in enumerate(classNames):
            classMappingFile.write(f'{className},{id}\n')
    return classMappingFileName


def generateDatasets():
    classNames1, trainAnnotations = generateDataset('dataset/train', 200)
    classNames2, valAnnotations = generateDataset('dataset/val', 50)
    classMapping = saveClassMapping('dataset', classNames1.union(classNames2))
    return trainAnnotations, valAnnotations, classMapping


def train(trainAnnotationsFile, classMappingFile, valAnnotationFile=None, batchSize=2,
          snapshot=None,
          weights=None,
          randomTransform=False):
    args = ['--workers=0',
            f'--batch-size={batchSize}']
    if snapshot:
        args.append(f'--snapshot={snapshot}')
    elif weights is not None:
        args.append(f'--weights={weights}')
    else:
        args.append('--weights=../../snapshots/ResNet-50-model.keras.h5')
    if randomTransform:
        args.append('--random-transform')
    args.extend(['csv',
                 trainAnnotationsFile,
                 classMappingFile
                 ])
    if valAnnotationFile is not None:
        args.append(f'--val-annotations={valAnnotationFile}')
    return trainRetina(args)


def main():
    # TODO: add noise
    # TODO: add ellipses

    trainAnnotations, valAnnotations, classMapping = generateDatasets()
    # trainAnnotations, valAnnotations, classMapping = 'dataset/train/annotations.csv', 'dataset/val/annotations.csv', 'dataset/class_mapping.csv'

    ret = train(trainAnnotations, classMapping, valAnnotations, batchSize=2,
                snapshot=None,  # './snapshots/resnet50_csv_08.h5'
                weights=None,  # '../../snapshots/resnet50_coco_best_v2.1.0.h5',
                randomTransform=True)
    # TODO: test ellipse detection (we train only on circles)


if __name__ == '__main__':
    main()
