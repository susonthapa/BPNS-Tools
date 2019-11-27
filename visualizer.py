import cv2
import os
import numpy as np
import xml.etree.ElementTree as parser
import colorsys

root = '/home/suson/AI/datasets/'
dataset_name = 'coco_train'


def random_colors(n, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / n, 1, brightness) for i in range(n)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    np.random.shuffle(colors)
    return colors


def visualize():
    cv2.startWindowThread()
    root_dir = os.path.join(root, dataset_name, 'validation')
    test_dir = os.path.join(root, dataset_name, 'validation', 'images')
    test_files = []
    # test_model.detect(os.path.join('/home/suson/AI/datasets/stanford-dogs-overfit/train/images/n02085620_7.jpg'))
    for file in os.listdir(os.fsencode(test_dir)):
        filename = os.fsdecode(file)
        test_files.append(os.path.join(test_dir, filename))

    test_files = np.array(test_files)
    # randomly shuffle the test images
    np.random.shuffle(test_files)
    # loop through each images in the list
    i = 0
    for file in test_files:
        image = cv2.imread(file)

        # Obtain the annotation file name from the image names
        annotation_name = os.path.join(root_dir, 'annotations', os.path.splitext(os.path.basename(file))[0] + '.xml')
        # parse the annotation file
        tree = parser.parse(annotation_name)

        # loop through all the object present in the annotations file
        # convert the height and width to the network accepted values
        objects = tree.findall('object')
        colors = random_colors(len(objects))
        index = 0
        for object in objects:
            box = object.find('bndbox')
            # compute the (x, y) co-ordinates
            x1 = int(box.find('xmin').text)
            y1 = int(box.find('ymin').text)
            x2 = int(box.find('xmax').text)
            y2 = int(box.find('ymax').text)
            color = tuple([rgb * 255 for rgb in colors[index]])
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
            cv2.putText(image, object.find('name').text, (x1 + 1, y1 + 8),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color, 1)

            index += 1
        cv2.imshow('Image', image)
        cv2.waitKey(0)


visualize()
