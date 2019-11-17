import json
from collections import defaultdict
import os
import urllib.request
from multiprocessing.dummy import Pool as ThreadPool
import xml.etree.ElementTree as xml_parser


parallel_threads = 10


class CoCoConverter(object):
    def __init__(self, json_path, dataset_root, classes, limit):
        # load the json file path
        # initialize the default dict to hold the annotations
        self.dataset = json.load(open(json_path, 'r'))
        print('Parsing Annotation File')
        self.img_to_annotations = defaultdict(list)
        self.img_id_to_img = {}
        self.categories = {}
        self.dataset_annotations = os.path.join(dataset_root, 'annotations')
        self.dataset_images = os.path.join(dataset_root, 'images')
        # obtain the category ids from the dataset
        category_ids = [cat['id'] for cat in self.dataset['categories'] if cat['name'] in classes]

        for category in self.dataset['categories']:
            self.categories[category['id']] = category
        print('Categories \n {}'.format(self.categories))

        # loop through the annotations in dataset
        count = 0
        for annotation in self.dataset['annotations']:
            # filter classes through the annotations
            if annotation['category_id'] in category_ids:
                self.img_to_annotations[annotation['image_id']].append(annotation)
                count += 1
            if len(self.img_to_annotations) >= limit:
                break

        for image in self.dataset['images']:
            self.img_id_to_img[image['id']] = image


        print('Finished Parsing Images, \n\tTotal Images -> {} \n\tTotal Annotations -> {}'.format(len(self.img_to_annotations), count))
        print('Generating Pascal VOC XML Format Annotations')
        # initialize multiple threads for processing the parsing
        pool = ThreadPool(parallel_threads)
        pool.starmap(self.downloader, self.img_to_annotations.items())
        pool.close()
        pool.join()

    def downloader(self, k, v):
        image_url = self.img_id_to_img[k]['coco_url']
        urllib.request.urlretrieve(image_url, os.path.join(self.dataset_images, str(k) + '.jpg'))
        print('Downloaded Image ID -> {}'.format(k))
        print('Generating Annotation Image ID -> {}'.format(k))
        # make xml annotation of the file
        annotation = xml_parser.Element('annotation')
        xml_parser.SubElement(annotation, 'filename').text = str(k) + 'jpg'

        size = xml_parser.SubElement(annotation, 'size')
        xml_parser.SubElement(size, 'width').text = str(self.img_id_to_img[k]['width'])
        xml_parser.SubElement(size, 'height').text = str(self.img_id_to_img[k]['height'])

        annotation_file = os.path.join(self.dataset_annotations, str(k) + '.xml')
        for detectionObject in v:
            objects = xml_parser.SubElement(annotation, 'object')
            xml_parser.SubElement(objects, 'name').text = self.categories[detectionObject['category_id']]['name']
            bnd_box = xml_parser.SubElement(objects, 'bndbox')

            xml_parser.SubElement(bnd_box, 'xmin').text = str(int(detectionObject['bbox'][0]))
            xml_parser.SubElement(bnd_box, 'ymin').text = str(int(detectionObject['bbox'][1]))
            xml_parser.SubElement(bnd_box, 'xmax').text = str(int(detectionObject['bbox'][0] + detectionObject['bbox'][2]))
            xml_parser.SubElement(bnd_box, 'ymax').text = str(int(detectionObject['bbox'][1] + detectionObject['bbox'][3]))

        tree = xml_parser.ElementTree(annotation)
        tree.write(annotation_file)
        print('Finished Generation For Image ID -> {}'.format(k))


def main():
    json_file = '/home/suson/AI/cocoapi/PythonAPI/coco_annotations/annotations/instances_val2017.json'
    classes = ['person']
    dataset_path = '/home/suson/AI/datasets/coco_test/train/'
    convertor = CoCoConverter(json_file, dataset_path, classes, 200)



if __name__ == '__main__':
    main()












