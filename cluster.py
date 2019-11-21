import random
import numpy as np
import os
import xml.etree.ElementTree as parser
import sys
import matplotlib.pyplot as plt
import json
import seaborn as sns
#sys.path.insert(0, '/home/suson/Projects/BPNS/bpns-model/')

sns.set()
cell_size = 7
root_dir = '/home/suson/AI/datasets/'
dataset_name = 'custom_bpns_224_dataset'
classes = ["person", "bus", "car", "bicycle", "motorcycle", "truck"]

current_palette = list(sns.xkcd_rgb.values())
# function to compute the anchors from the annotations files
def read_anchors():
    print('Parsing Label For -> Anchor Box Generation!')
    annotations_anchor = []
    normalized_wh_anchors = []
    labels_dict = {}
    file_path = os.path.join(root_dir, dataset_name, 'train', 'annotations')
    # loop through all the annotations in the directory
    for file in os.listdir(os.fsencode(file_path)):
        img = {'object': []}
        filename = os.fsdecode(file)

        img['filename'] = filename
        tree = parser.parse(os.path.join(file_path, filename))
        img_size = tree.find('size')
        img_width = float(img_size.find('width').text)
        img_height = float(img_size.find('height').text)

        img['width'] = img_width
        img['height'] = img_height

        cell_height = 1.0 * (img_height / cell_size)
        cell_width = 1.0 * (img_width / cell_size)
        objects = tree.findall('object')
        
        for object in objects:
            box = object.find('bndbox')
            # compute the (x, y) co-ordinates
            x1 = float(box.find('xmin').text)
            y1 = float(box.find('ymin').text)
            x2 = float(box.find('xmax').text)
            y2 = float(box.find('ymax').text)


            obj = {}

            obj['name'] = object.find('name').text
            obj['xmin'] = x1
            obj['ymin'] = y1
            obj['xmax'] = x2
            obj['ymax'] = y2

            if obj['name'] in labels_dict:
                labels_dict[obj['name']] += 1
            else:
                labels_dict[obj['name']] = 1

            
            img['object'] += [obj]

            relative_w = (x2 - x1) / cell_width
            relative_h = (y2 - y1) / cell_height

            normalized_wh_anchors.append([relative_w, relative_h])
        annotations_anchor.append(img)
    return annotations_anchor, labels_dict, normalized_wh_anchors


# use jaccard index to as a similarity matrix
def compute_similarity(anchor, centroids):
    width, height = anchor
    similarities = []
    # loop through all the centroids
    for centroid in centroids:
        c_width, c_height = centroid
        # compute the intersection of the centroid box and given anchor box i.e (AnB)/(AuB)
        similarity = (min(width, c_width) * min(height, c_height)) / (
                width * height + c_height * c_width - min(width, c_width) * min(height, c_height) + 1e-10)
        similarities.append(similarity)
    return np.array(similarities)


def generate_anchors(annotations_anchor, num_anchors):
    total_anchor = len(annotations_anchor)
    # old assignment
    prev_assignments = np.ones(total_anchor) * (-1)
    old_distance = np.zeros([total_anchor, num_anchors])
    # randomly select any anchors indices
    indices = [random.randrange(total_anchor) for i in range(num_anchors)]
    # randomly set the starting anchors
    centroids = annotations_anchor[indices]
    iteration = 0
    # loop until the centroids doesn't changes
    while True:
        distances = []
        iteration += 1
        for i in range(total_anchor):
            distance = 1 - compute_similarity(annotations_anchor[i], centroids)
            distances.append(distance)
        distances = np.array(distances)

        print("\tIterations {} : Delta = {}".format(iteration, np.sum(np.abs(old_distance - distances))))
        # compute the centroids with minimum distance with the previous centroids
        # i.e the assignment of the anchors to the respective centroid
        assignments = np.argmin(distances, axis=1)
        # check if the previous points that we assigned to the centroid are same, i.e we have found the clusters
        if (assignments == prev_assignments).all():
            # we have found the cluster, so compute the jaccard_metric
            total_jaccard_metric = np.sum(distances[np.arange(distances.shape[0]), assignments]) / total_anchor
            return total_jaccard_metric, centroids, distances, assignments
        centroids_sum = np.zeros([num_anchors, 2], np.float)
        for i in range(total_anchor):
            centroids_sum[assignments[i]] += annotations_anchor[i]

        for j in range(num_anchors):
            centroids[j] = centroids_sum[j] / (np.sum(assignments == j) + 1e-6)

        prev_assignments = assignments.copy()
        old_distance = distances.copy()


def plot_clusters(axes, centroids, nearest_centroid, total_distance, normalized_wh):
    for i in np.unique(nearest_centroid):
        # find the points that belongs to the particular cluster
        pick = nearest_centroid==i
        palette = current_palette[i]
        # plt.rc('font', size=8)
        axes.plot(normalized_wh[pick, 0], normalized_wh[pick, 1], 'p', color=palette,
            alpha=0.5, label='Cluster = {}, Points = {:6.0f}'.format(i, np.sum(pick)))
        axes.legend(loc='lower right', title='Mean Distance = {:6.0f}'.format(total_distance))
        axes.annotate('C{}'.format(i), xy=(centroids[i, 0] / cell_size, centroids[i, 1] / cell_size), va='center', ha='center', color='red')
        axes.set_title('Clusters(K={})'.format(centroids.shape[0]))
        axes.set_xlabel('Normalized Width')
        axes.set_ylabel('Normalized Height')



def find_clusters():
    kmax = 6
    annotations_anchor, total_classes, normalized_wh_anchors = read_anchors()

    #print(json.dumps(annotations_anchor[:2], sort_keys=True, indent=4))
    #print(total_classes)

    # visualize the total classes
    y_position = np.arange(len(total_classes))
    fig = plt.figure(figsize=(10, 10))
    axis = fig.add_subplot(1, 1, 1)
    axis.barh(y_position, list(total_classes.values()))
    axis.set_yticks(y_position)
    axis.set_yticklabels(list(total_classes.keys()))
    axis.set_title('Total Objects = {}\n Total Images = {}'.format(np.sum(list(total_classes.values())), len(annotations_anchor)))
    #plt.show()
    plt.savefig('images_per_class.jpg', dpi=100)

    # visualize the clusters
    normalized_wh_anchors = np.array(normalized_wh_anchors)
    normalized_wh = normalized_wh_anchors / cell_size
    plt.figure(figsize=(10, 10))
    plt.scatter(normalized_wh[:, 0], normalized_wh[:, 1], alpha=0.1)
    plt.title('Clusters')
    plt.xlabel('Normalized Width')
    plt.ylabel('Normalized Height')
    #plt.show()
    plt.savefig('image_distribution.jpg', dpi=100)

    # initialize the centroids
    clusters = np.zeros([kmax])
    # create index as total number of clusters to find
    x = np.arange(1, kmax + 1)
    fig = plt.figure(figsize=(50, 50))
    results = {}
    count = 1
    for j in x:
        print('Initializing K Means Clustering!')
        print('\n\tNumber Of Clusters => {}'.format(j))
        outputs = []
        # compute the total distance and centroids from the given points
        total_distance, centroids, distances, assignments = generate_anchors(normalized_wh_anchors, j)
        clusters[j-1] = total_distance
        result = {
            'centroids': centroids,
            'nearest_centroid': assignments,
            'distances': distances,
            'total_distance': total_distance
        }
        results[j] = result
        
        print('\nTotal Distance => {}, Anchors => {}'.format(total_distance, np.reshape(
            centroids, [-1])))

    for j in x:
        result = results[j]
        centroids = result['centroids']
        nearest_centroid = result['nearest_centroid']
        total_distance = result['total_distance']

        axes = fig.add_subplot(kmax / 2, 2, count)
        plot_clusters(axes, centroids, nearest_centroid, total_distance, normalized_wh)
        count += 1
    plt.savefig('clustering_output.jpg', dpi=100)
    #plt.show()
    '''
    # add the distances and centroids, so that they can be compared
    outputs.append([total_distance, centroids])

    # convert to numpy
    outputs = np.array(outputs)
    # find the minimum distances
    best_cluster_index = np.argmin(outputs[:, 0], axis=0)
    clusters[j-1] = outputs[best_cluster_index, 0]
    print('No Of Clusters => {}, Minimum Distances Total => {}, Anchors => {}'.format(j, outputs[best_cluster_index, 0], 
        np.reshape(outputs[best_cluster_index, 1], [-1])))
    '''

    # visualize the results
    plt.figure(figsize=(10, 10))
    plt.plot(x, clusters, label='Elbow Curve')
    plt.ylim(0, 1.0)
    plt.xticks(x)
    plt.ylabel('No Of Clusters')
    plt.xlabel('Jaccard Mean Distance')
    plt.savefig('elbow_curve.jpg', dpi=100)
    #plt.show()


find_clusters()
