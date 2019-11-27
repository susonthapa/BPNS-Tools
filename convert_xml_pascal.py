import os
import xml.etree.ElementTree as xml_parser
import shutil
import argparse
import cv2
import re
from os.path import join, splitext

desired_size = 224
classes = dict({"person": 0, "bus": 0, "car": 0, "bicycle": 0, "motorcycle": 0, "motorbike": 0, "truck": 0})

vehicle_class = ['bus', 'car', 'truck']
bike_class = ['motorbike', 'bicycle', 'motorcycle']


# function to convert txt annotation from imagenet to pascal voc format
def txt_to_xml(src_annotation, src_img, dest_annotation, dest_img):
    # index for naming
    index = 0
    # list all the annotations file from the source directory
    for file in os.listdir(os.fsencode(src_annotation)):
        # get the file name
        filename = os.fsdecode(file)
        path = join(src_annotation, filename)
        print('Parsing File {}'.format(filename))
        # read the image file to determine the height and width of the images
        image_name = splitext(filename)[0] + '.jpg'
        src_image = join(src_img, image_name)
        image = cv2.imread(src_image)
        height, width = image.shape[0:2]


        '''
        height_ratio = float(desired_size) / height
        width_ratio = float(desired_size) / width
        image = cv2.resize(image, (desired_size, desired_size))


        '''
        # parse the existing file
        old_xml = xml_parser.parse(path)
        is_object_found = False
        # loop through each objects in the xml file
        old_objects = old_xml.findall('object')
        class_name = ''

        annotation = xml_parser.Element('annotation')
        # store the image sizes
        size = xml_parser.SubElement(annotation, 'size')

        '''
        xml_parser.SubElement(size, 'width').text = str(desired_size)
        xml_parser.SubElement(size, 'height').text = str(desired_size)
        '''

        xml_parser.SubElement(size, 'width').text = str(width)
        xml_parser.SubElement(size, 'height').text = str(height)

        for object in old_objects:
            # check if the object is present in our required dataset classes
            if object.find('name').text not in classes.keys():
                continue

            # store the information of the objects within the image
            box = object.find('bndbox')

            '''
            xmin = int(float(box.find('xmin').text) * width_ratio)
            ymin = int(float(box.find('ymin').text) * height_ratio)
            xmax = int(float(box.find('xmax').text) * width_ratio)
            ymax = int(float(box.find('ymax').text) * height_ratio)
            '''


            xmin = int(float(box.find('xmin').text))
            ymin = int(float(box.find('ymin').text))
            xmax = int(float(box.find('xmax').text))
            ymax = int(float(box.find('ymax').text))

            # compute the area of the object w.r.t to image
            if (((xmax - xmin) * (ymax - ymin)) / float(height * width)) < 0.0204:
                print('Object Too Small, Skipping!')
                continue

            '''
            # check if the object is too small
            if (xmax - xmin) * (ymax - ymin) < 1024:
                print('Annotation Too Small, Skipping!')
                continue

            '''
            objects = xml_parser.SubElement(annotation, 'object')
            class_name = object.find('name').text
            if class_name in vehicle_class:
                class_name = 'vehicle'

            elif class_name in bike_class:
                class_name = 'bike'

            print('\n\t\t Object => {}'.format(class_name))
            xml_parser.SubElement(objects, 'name').text = class_name

            
            bnd_box = xml_parser.SubElement(objects, 'bndbox')
            is_object_found = True
            xml_parser.SubElement(bnd_box, 'xmin').text = str(xmin)
            xml_parser.SubElement(bnd_box, 'ymin').text = str(ymin)
            xml_parser.SubElement(bnd_box, 'xmax').text = str(xmax)
            xml_parser.SubElement(bnd_box, 'ymax').text = str(ymax)
        # store the file name
        image_name = class_name + str(index) + '.jpg'
        xml_parser.SubElement(annotation, 'filename').text = image_name

        # check if the xml file is valid
        if is_object_found:
            # save the xml file
            filename = class_name + str(index) + '.xml'
            print('\t Image => {}'.format(image_name))
            print('\t Annotation => {}'.format(filename))
            tree = xml_parser.ElementTree(annotation)
            tree.write(join(dest_annotation, filename))
            # copy the image file
            cv2.imwrite(join(dest_img, image_name), image)
            index += 1
        else:
            print('No Objects Of Class Found, Skipping Image!')

def main():
    #txt_to_xml('/home/suson/AI/datasets/final_coco_pascal_dataset/train/annotations/', '/home/suson/AI/datasets/final_coco_pascal_dataset/train/images/', '/home/suson/AI/datasets/coco-pascal-darknet-less-class/train/annotations/', '/home/suson/AI/datasets/coco-pascal-darknet-less-class/train/images/')
    txt_to_xml('/home/suson/AI/datasets/coco_train/validation/annotations/', '/home/suson/AI/datasets/coco_train/validation/images/', '/home/suson/AI/datasets/coco-pascal-darknet-less-class/validation/annotations/', '/home/suson/AI/datasets/coco-pascal-darknet-less-class/validation/images/')
    #txt_to_xml('/home/suson/AI/datasets/final_coco_pascal_dataset/validation/annotations/', '/home/suson/AI/datasets/final_coco_pascal_dataset/validation/images/', '/home/suson/AI/datasets/custom_coco_no_resize/validation/annotations/', '/home/suson/AI/datasets/custom_coco_no_resize/validation/images/')


if __name__ == '__main__':
    main()
