import json
from collections import defaultdict
import os
import urllib.request
from multiprocessing.dummy import Pool as ThreadPool
import xml.etree.ElementTree as xml_parser
import shutil

parallel_threads = 25


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

        # create individual counter for annotations of each classes
        self.classes_counter = {}
        for i in range(len(category_ids)):
            self.classes_counter[category_ids[i]] = 0

        for category in self.dataset['categories']:
            self.categories[category['id']] = category

        # loop through the annotations in dataset
        count = 0
        for annotation in self.dataset['annotations']:
            # filter classes through the annotations
            if annotation['category_id'] in category_ids:
                self.img_to_annotations[annotation['image_id']].append(annotation)
                self.classes_counter[annotation['category_id']] += 1
                
                if self.classes_counter[annotation['category_id']] >= limit:
                    category_ids.remove(annotation['category_id'])

        for image in self.dataset['images']:
            self.img_id_to_img[image['id']] = image

        #for k, v in self.img_to_annotations.items():
        #    if not os.path.isfile(os.path.join(self.dataset_images, str(k) + '.jpg')):
         #       print('Image File doesnot exists {}'.format(k))
          #  if not os.path.isfile(os.path.join(self.dataset_annotations, str(k) + '.xml')):
           #     print('Annotations File doesnot exists {}'.format(k))


        print('Finished Parsing Images, \n\tTotal Images -> {}'.format(len(self.img_to_annotations)))
        print('\n\tTotal Annotations:')
        for k, v in self.classes_counter.items():
            print('\n\t\t{} -> {}'.format(self.categories[k]['name'], v))
        input('Press any key to continue')
        print('Generating Pascal VOC XML Format Annotations')
        # initialize multiple threads for processing the parsing
        pool = ThreadPool(parallel_threads)
        pool.starmap(self.downloader, self.img_to_annotations.items())
        pool.close()
        pool.join()

    def downloader(self, k, v):
        image_url = self.img_id_to_img[k]['coco_url']
        # check if the file exists
        #if os.path.isfile(os.path.join(self.dataset_images, str(k) + '.jpg')):
        #    print('Image {}.jpg already exists, skipping image download!'.format(k))
        #    if os.path.isfile(os.path.join(self.dataset_annotations, str(k) + '.xml')):
        #        print('Annotations {}.jpg already exists, skipping annotation generation!'.format(k))
                #shutil.move(os.path.join(self.dataset_images, str(k) + '.jpg'), os.path.join('/home/suson/AI/datasets/coco_train/train/images/', str(k) + '.jpg'))
                #shutil.move(os.path.join(self.dataset_annotations, str(k) + '.xml'), os.path.join('/home/suson/AI/datasets/coco_train/train/annotations/', str(k) + '.xml'))
        #        return
        #    else:
        #        #print('Image {} does\'t have corresponding Annotation(Image file may be corrupted!). Downloading Again!'.format(k))
        urllib.request.urlretrieve(image_url, os.path.join(self.dataset_images, str(k) + '.jpg'))
        print('Downloaded Image ID -> {}'.format(k))
        print('Generating Annotation Image ID -> {}'.format(k))
        # make xml annotation of the file
        annotation = xml_parser.Element('annotation')
        xml_parser.SubElement(annotation, 'filename').text = str(k) + '.jpg'

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
    classes = ['person', 'bus', 'car', 'bicycle', 'motorcycle', 'truck']
    dataset_path = '/home/suson/AI/datasets/coco_train/validation/'
    convertor = CoCoConverter(json_file, dataset_path, classes, 5000)



if __name__ == '__main__':
    main()












