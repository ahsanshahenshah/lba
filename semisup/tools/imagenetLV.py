import os
import yaml
import tensorflow as tf
import numpy as np
from lv.nn.augmentation.common_augmentation import augment_image
from semisup.tools import data_dirs
from sklearn.model_selection import train_test_split
name_d = 'imagenetLV_10'
detectors_dict = {}
this_dir = os.path.dirname(__file__)
detector_config = os.path.join(this_dir, 'classifiers.yaml')

NUM_LABELS = 10
IMAGE_SHAPE = [96, 96, 3]

def read_detectors():
    with open(detector_config,'r') as f:
        detectors = yaml.load(f)

    for det in detectors:
        detectors_dict[det['name']] = det

def get_detector(name):
    if detectors_dict.get(name) is not None:
        return detectors_dict[name]
    else:
        Exception("Detector:{} not configured in {}".format(name, detector_config))

def get_data(name):
    read_detectors()

    d = get_detector(name_d)
    print d
    print d['dataset_dir']
    all_labels = []
    X_train, y_train, X_test, y_test = [], [], [], []
    for cls_ind, cls in enumerate(d['classes']):
        cls_dir = os.path.join(d['dataset_dir'], cls)
        img_paths = [os.path.join(cls_dir, file_name) for file_name in sorted(os.listdir(cls_dir))]
        labels = len(img_paths) * [cls_ind]  # for example 3 * [10] = [10, 10, 10]
        all_labels += labels
        X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(img_paths, labels, random_state=1, test_size=0.10)
        X_train += X_train_tmp
        y_train += y_train_tmp
        X_test += X_test_tmp
        y_test += y_test_tmp

    if name == 'train' or name == 'unlabeled':
        print ('number of training images',len(X_train))
        return np.array(X_train),np.array(y_train)
    elif name == 'test':
        print ('number of test images', len(y_test))
        return np.array(X_test),np.array(y_test)

read_detectors()

d = get_detector(name_d)


CLASSES = d['classes']























'''
def create_datasets(dataset_dir, classes, input_shape, sess, augment=False, shuffle=False,
                    test_size=None, batch_size=32, random_seed=None, grayscale=False):
    # Reads an image from a file, decodes it into a dense tensor, and resizes it
    # to a fixed shape.
    args = dict()
    if test_size == 1.0:
        args['train_size'] = 0.0
    else:
        args['test_size'] = test_size

    def _parse_function(filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.cast(tf.image.decode_jpeg(image_string, channels=3), tf.float32)
        image_resized = tf.image.resize_images(image_decoded, (input_shape[0], input_shape[1]))
        if grayscale:
            image_resized = tf.image.grayscale_to_rgb(image_resized)
        image_resized = image_resized * tf.constant(1. / 255.)
        return image_resized

    def _parse_function_train(filename, label):
        image_resized = _parse_function(filename)

        if augment: image_resized = augment_image(image_resized, input_shape, random_seed=random_seed)
        return image_resized, tf.one_hot(label, depth=len(classes))

    def _parse_function_test(filename, label):
        image_resized = _parse_function(filename)

        return image_resized, tf.one_hot(label, depth=len(classes))

    X_train, y_train, X_test, y_test = [], [], [], []

    # this is needed to compute class_weights
    all_labels = []

    for cls_ind, cls in enumerate(classes):
        cls_dir = os.path.join(dataset_dir, cls)
        img_paths = [os.path.join(cls_dir, file_name) for file_name in os.listdir(cls_dir)]
        labels = len(img_paths) * [cls_ind]  # for example 3 * [10] = [10, 10, 10]
        all_labels += labels

        X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(img_paths, labels, random_state=random_seed,
                                                                            **args)
        X_train += X_train_tmp
        y_train += y_train_tmp
        X_test += X_test_tmp
        y_test += y_test_tmp

    #class_weights = get_class_weights(all_labels)

    if test_size < 1.0:
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))

        if shuffle:
            buffersize = min(len(X_train), 10000)
            train_dataset = train_dataset.shuffle(buffer_size=buffersize, seed=random_seed)

        train_dataset = train_dataset.map(_parse_function_train, num_parallel_calls=5).repeat().batch(batch_size)
        train_dataset = train_dataset.prefetch(buffer_size=200)
        # This takes less RAM memory, but speed is quite equal (prefetch_to_device little faster)
        # train_dataset = train_dataset.apply(tf.contrib.data.prefetch_to_device('/device:GPU:0', buffer_size=200))
        iterator_train = train_dataset.make_initializable_iterator()
        next_element_train = iterator_train.get_next()

    if test_size != 0.0:
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_dataset = test_dataset.map(_parse_function_test, num_parallel_calls=5).repeat().batch(batch_size)
        iterator_test = test_dataset.make_initializable_iterator()
        next_element_test = iterator_test.get_next()

    def gen_train():
        sess.run(iterator_train.initializer)
        while True:
            try:
                nxb, nxl = sess.run(next_element_train)
                yield nxb, nxl
            except tf.errors.OutOfRangeError:
                sess.run(iterator_train.initializer)  # this should never happen anyway, because we repeat indefinitely

    def gen_test():
        sess.run(iterator_test.initializer)
        while True:
            try:
                nxb, nxl = sess.run(next_element_test)
                yield nxb, nxl
            except tf.errors.OutOfRangeError:
                sess.run(iterator_test.initializer)

    return gen_train(), gen_test(), X_train, X_test, class_weights



'''
