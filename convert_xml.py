import os
import xml.etree.ElementTree as xml_parser
import shutil
import argparse
import cv2
import re
from os.path import join, splitext

desired_size = 224


def convert():
    for file in os.listdir(os.fsencode('/home/suson/AI/datasets/stanford-dog/train/annotations/')):
        filename = os.fsdecode(file)
        path = join('/home/suson/AI/datasets/stanford-dog/train/annotations/', filename)
        print(path)
        tree = xml_parser.parse(path)
        root = tree.getroot()
        objects = tree.findall('object')
        for object in objects:
            name = object.find('name')
            name.text = 'dog'

        tree = xml_parser.ElementTree(root)
        tree.write(path, xml_declaration=True)
        # shutil.move(path, path + '.xml')


# function to convert txt annotation from imagenet to pascal voc format
def txt_to_xml(src_annotation, src_img, dest_annotation, dest_img, class_name, indices, limit, sensitivity=0.0):
    # index for naming
    index = indices
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

        height_ratio = float(desired_size) / height
        width_ratio = float(desired_size) / width

        # resize the image with padding maintaining the aspect ratio
        # ratio = float(desired_size) / max(image.shape[0:2])
        # # compute new size
        # new_height = int(ratio * height)
        # new_width = int(ratio * width)
        # resize the image
        # image = cv2.resize(image, (new_width, new_height))
        image = cv2.resize(image, (desired_size, desired_size))
        # compute the change in size of the smallest side
        # delta_width = desired_size - new_width
        # delta_height = desired_size - new_height
        # # distribute the delta symmetrically
        # top, bottom = delta_height // 2, delta_height - (delta_height // 2)
        # left, right = delta_width // 2, delta_width - (delta_width // 2)
        # padding the images with zeros
        # image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # generate new xml file
        annotation = xml_parser.Element('annotation')
        # store the file name
        image_name = class_name + str(index) + '.jpg'
        xml_parser.SubElement(annotation, 'filename').text = image_name
        # store the image sizes
        size = xml_parser.SubElement(annotation, 'size')
        xml_parser.SubElement(size, 'width').text = str(desired_size)
        xml_parser.SubElement(size, 'height').text = str(desired_size)
        with open(path, 'r') as annotation_file:
            is_object_found = False
            # loop through each line
            for f in annotation_file:
                # store the information of the objects within the image
                objects = xml_parser.SubElement(annotation, 'object')
                # split the file on spaces
                content = f.split()
                print('\t Annotation => {}'.format(content))
                # parse the contents in the line
                xml_parser.SubElement(objects, 'name').text = class_name
                bnd_box = xml_parser.SubElement(objects, 'bndbox')
                # compute the offset of the bounding box
                x = 0
                # search if the current item contains string or not
                while re.search('[a-zA-Z]', content[x]):
                    x += 1
                # xml_parser.SubElement(bnd_box, 'xmin').text = str(int(float(content[x]) * ratio + delta_width // 2))
                # xml_parser.SubElement(bnd_box, 'ymin').text = str(int(float(content[x + 1]) * ratio + delta_height // 2))
                # xml_parser.SubElement(bnd_box, 'xmax').text = str(int(float(content[x + 2]) * ratio + delta_width // 2))
                # xml_parser.SubElement(bnd_box, 'ymax').text = str(int(float(content[x + 3]) * ratio + delta_height // 2))
                # check the sensitivity parameter, to control the size of the objects with respect to the image size
                object_ratio = (float(content[x + 2]) - float(content[x])) * (float(content[x + 3]) - float(content[x + 1])) / (height * width)
                if object_ratio < sensitivity:
                    # the object is too small for our model
                    continue
                is_object_found = True
                xml_parser.SubElement(bnd_box, 'xmin').text = str(int(float(content[x]) * width_ratio))
                xml_parser.SubElement(bnd_box, 'ymin').text = str(int(float(content[x + 1]) * height_ratio))
                xml_parser.SubElement(bnd_box, 'xmax').text = str(int(float(content[x + 2]) * width_ratio))
                xml_parser.SubElement(bnd_box, 'ymax').text = str(int(float(content[x + 3]) * height_ratio))
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
            else:
                print('\t Skipping Images, Object Too Small!!!')

            index += 1
            if index >= limit:
                break


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--src_annotation', default=None, type=str)
    arg_parser.add_argument('--src_img', default='', type=str)
    arg_parser.add_argument('--dest_root_dir', default='', type=str)
    arg_parser.add_argument('--dest_img', default='', type=str)
    arg_parser.add_argument('--class_name', default='', type=str)
    arg_parser.add_argument('--index', default=0, type=int)
    arg_parser.add_argument('--limit', default=2500, type=int)
    arg_parser.add_argument('--use_code', default=False, type=bool)
    arg_parser.add_argument('--sensitivity', default=0, type=float)
    args = arg_parser.parse_args()
    # use direct code for conversion
    if args.use_code:
        print('Parsing From Direct Code, This will take some time!!!!\n\n\n')
        root_dir = '/home/suson/AI/datasets/custom_bpns_224_dataset/'
        dest_annotation = join(root_dir, 'train', 'annotations')
        dest_img = join(root_dir, 'train', 'images')
        max_limit_train = 2500
        max_limit_test = 250
        print('Parsing For Training!!!!\n\n\n')
        
        # Conversion For classes
        src_img = '/home/suson/AI/datasets/OID/train/Bicycle'
        txt_to_xml(join(src_img, 'Label'), src_img, dest_annotation, dest_img, 'bicycle', 0, int(max_limit_train))
        # Conversion For classes
        src_img = '/home/suson/AI/datasets/OID/train/Motorcycle'
        txt_to_xml(join(src_img, 'Label'), src_img, dest_annotation, dest_img, 'motorcycle', 0, int(max_limit_train))
        # Conversion For classes
        src_img = '/home/suson/AI/datasets/OID/train/Person'
        txt_to_xml(join(src_img, 'Label'), src_img, dest_annotation, dest_img, 'person', 0, int(max_limit_train * 0.25))
        # Conversion For classes
        src_img = '/home/suson/AI/datasets/OID/train/Boy'
        txt_to_xml(join(src_img, 'Label'), src_img, dest_annotation, dest_img, 'person', int(max_limit_train * 0.25), int(max_limit_train * 0.5))
        # Conversion For classes
        src_img = '/home/suson/AI/datasets/OID/train/Woman'
        txt_to_xml(join(src_img, 'Label'), src_img, dest_annotation, dest_img, 'person', int(max_limit_train * 0.25), int(max_limit_train * 0.75))
        # Conversion For classes
        src_img = '/home/suson/AI/datasets/OID/train/Girl'
        txt_to_xml(join(src_img, 'Label'), src_img, dest_annotation, dest_img, 'person', int(max_limit_train * 0.75), int(max_limit_train))
        # Conversion For classes
        src_img = '/home/suson/AI/datasets/OID/train/Car'
        txt_to_xml(join(src_img, 'Label'), src_img, dest_annotation, dest_img, 'car', 0, int(max_limit_train))
        # Conversion For classes
        src_img = '/home/suson/AI/datasets/OID/train/Bus'
        txt_to_xml(join(src_img, 'Label'), src_img, dest_annotation, dest_img, 'bus', 0, int(max_limit_train))
        # Conversion For classes
        src_img = '/home/suson/AI/datasets/OID/train/Truck'
        txt_to_xml(join(src_img, 'Label'), src_img, dest_annotation, dest_img, 'truck', 0, int(max_limit_train))
        
        print('Parsing For Testing!!!!\n\n\n')
        # dataset for validation
        dest_annotation = join(root_dir, 'validation', 'annotations')
        dest_img = join(root_dir, 'validation', 'images')
        # Conversion For classes
        src_img = '/home/suson/AI/datasets/OID/test/Bicycle'
        txt_to_xml(join(src_img, 'Label'), src_img, dest_annotation, dest_img, 'bicycle', 0, int(max_limit_test))
        src_img = '/home/suson/AI/datasets/OID/train/Motorcycle'
        txt_to_xml(join(src_img, 'Label'), src_img, dest_annotation, dest_img, 'motorcycle', 0, int(max_limit_test))
        # Conversion For classes
        src_img = '/home/suson/AI/datasets/OID/test/Person'
        txt_to_xml(join(src_img, 'Label'), src_img, dest_annotation, dest_img, 'person', 0, int(max_limit_test * 0.25))
        # Conversion For classes
        src_img = '/home/suson/AI/datasets/OID/test/Boy'
        txt_to_xml(join(src_img, 'Label'), src_img, dest_annotation, dest_img, 'person', int(max_limit_test * 0.25), int(max_limit_test * 0.5))
        # Conversion For classes
        src_img = '/home/suson/AI/datasets/OID/test/Woman'
        txt_to_xml(join(src_img, 'Label'), src_img, dest_annotation, dest_img, 'person', int(max_limit_test * 0.5), int(max_limit_test * 0.75))
        # Conversion For classes
        src_img = '/home/suson/AI/datasets/OID/test/Girl'
        txt_to_xml(join(src_img, 'Label'), src_img, dest_annotation, dest_img, 'person', int(max_limit_test * 0.75), max_limit_test)
        # Conversion For classes
        src_img = '/home/suson/AI/datasets/OID/test/Car'
        txt_to_xml(join(src_img, 'Label'), src_img, dest_annotation, dest_img, 'car', 0, int(max_limit_test))
        # Conversion For classes
        src_img = '/home/suson/AI/datasets/OID/test/Bus'
        txt_to_xml(join(src_img, 'Label'), src_img, dest_annotation, dest_img, 'bus', 0, int(max_limit_test))
        # Conversion For classes
        src_img = '/home/suson/AI/datasets/OID/test/Truck'
        txt_to_xml(join(src_img, 'Label'), src_img, dest_annotation, dest_img, 'truck', 0, int(max_limit_test))
    else:
        if args.src_annotation is None:
            args.src_annotation = join(args.src_img, 'Label')
        dest_annotation = join(args.dest_root_dir, 'annotations')
        dest_img = join(args.dest_root_dir, 'images')

        # call the function to perform the conversion
        txt_to_xml(args.src_annotation, args.src_img, dest_annotation, dest_img, args.class_name, args.index,
                   args.limit, args.sensitivity)


if __name__ == '__main__':
    main()
