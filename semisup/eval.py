#! /usr/bin/env python
"""
Copyright 2016 Google Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Association-based semi-supervised eval module.

This script defines the evaluation loop that works with the training loop
from train.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from functools import partial
from importlib import import_module


import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
#plt.tight_layout()
import tfplot
import seaborn.apionly as sns

import semisup
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.python import debug as tf_debug


FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'imagenetLV', 'Which dataset to work on.')

flags.DEFINE_string('architecture', 'stl10_model', 'Which dataset to work on.')

flags.DEFINE_integer('eval_batch_size', 500, 'Batch size for eval loop.')

flags.DEFINE_integer('new_size', 0, 'If > 0, resize image to this width/height.'
                                    'Needs to match size used for training.')

flags.DEFINE_integer('emb_size', 128,
                     'Size of the embeddings to learn.')

flags.DEFINE_integer('eval_interval_secs', 10,
                     'How many seconds between executions of the eval loop.')

flags.DEFINE_string('logdir', '/data/semisup/opt_reproduce/imagenetLV_super_10x10_usup_4000_w_08_v_02_l_10_usb_200_supb_10_embs_128_stl10_model_seed_10_clrexp2_lr_envelope_aug_all_00_bnemb_5',
                    'Where the checkpoints are stored '
                    'and eval events will be written to.')

flags.DEFINE_string('master', '',
                    'BNS name of the TensorFlow master to use.')

flags.DEFINE_integer('timeout', 1200,
                     'The maximum amount of time to wait between checkpoints. '
                     'If left as `None`, then the process will wait '
                     'indefinitely.')

flags.DEFINE_bool('augmentation', False,
                  'Apply data augmentation during training.')

def cp(a):
    print(a)
    return a

def main(_):
    # Get dataset-related toolbox.
    dataset_tools = import_module('semisup.tools.' + FLAGS.dataset)
    architecture = getattr(semisup.architectures, FLAGS.architecture)

    num_labels = dataset_tools.NUM_LABELS
    image_shape = dataset_tools.IMAGE_SHAPE
    test_images, test_labels = dataset_tools.get_data('test')
    print (test_images)

    graph = tf.Graph()
    with graph.as_default():

        # Set up input pipeline.
        #image, label = tf.train.slice_input_producer([test_images, test_labels])
        #images, labels = tf.train.batch(
        #    [image, label], batch_size=FLAGS.eval_batch_size)
        images, labels = semisup.create_input(test_images,test_labels,FLAGS.eval_batch_size)

        images = tf.cast(images, tf.float32)
        labels = tf.cast(labels, tf.int64)

        # Reshape if necessary.
        if FLAGS.new_size > 0:
            new_shape = [FLAGS.new_size, FLAGS.new_size, 3]
        else:
            new_shape = None

        if FLAGS.augmentation:
            # TODO(haeusser) generalize augmentation
            def _random_invert(inputs1, _):
                inputs = tf.cast(inputs1, tf.float32)
                inputs = tf.image.adjust_brightness(inputs, tf.random_uniform((1, 1), 0.0, 0.5))
                inputs = tf.image.random_contrast(inputs, 0.3, 1)
                # inputs = tf.image.per_image_standardization(inputs)
                inputs = tf.image.random_hue(inputs, 0.05)
                inputs = tf.image.random_saturation(inputs, 0.5, 1.1)

                def f1(): return tf.abs(inputs)  # annotations

                def f2(): return tf.abs(inputs1)

                return tf.cond(tf.less(tf.random_uniform([], 0.0, 1), 0.5), f1, f2)

            augmentation_function = _random_invert
        else:
            augmentation_function = None

        # Create function that defines the network.
        model_function = partial(
            architecture,
            is_training=False,
            new_shape=new_shape,
            img_shape=image_shape,
            augmentation_function=augmentation_function,
            image_summary=False,
            emb_size=FLAGS.emb_size)


        # Set up semisup model.
        model = semisup.SemisupModel(
            model_function,
            num_labels,
            image_shape,
            test_in=images)

        # Add moving average variables.
        for var in tf.get_collection('moving_vars'):
            tf.add_to_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES, var)
        for var in slim.get_model_variables():
            tf.add_to_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES, var)

        # Get prediction tensor from semisup model.
        predictions = tf.argmax(model.test_logit, 1)
        cmatrix = tf.confusion_matrix(labels,predictions,num_labels)
        # Accuracy metric for summaries.
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        })
        for name, value in names_to_values.iteritems():
            tf.summary.scalar(name, value)
        confusion_image = tf.reshape(tf.cast(cmatrix, tf.float32),
                                     [1,10, 10, 1])
        tf.summary.tensor_summary('cmatrix', cmatrix)
        tf.summary.image('confusion matrix',confusion_image)
        tf_heatmap = tfplot.wrap_axesplot(sns.heatmap, figsize=(7,5), cbar=False, annot=True,yticklabels=dataset_tools.CLASSES, cmap='jet')
        tf.summary.image("heat_maps", tf.reshape(tf_heatmap(cmatrix),[1,500,700,4]))
        # Run the actual evaluation loop.
        num_batches = math.ceil(len(test_labels) / float(FLAGS.eval_batch_size))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        slim.evaluation.evaluation_loop(
            master=FLAGS.master,
            checkpoint_dir=FLAGS.logdir + '/train',
            logdir=FLAGS.logdir + '/eval',
            num_evals=num_batches,
            eval_op=tf.Print(list(names_to_updates.values()),[confusion_image], message="cmatrix:", summarize=500),
            eval_interval_secs=FLAGS.eval_interval_secs,
            session_config=config,
            timeout=FLAGS.timeout,
            #hooks=[tf_debug.LocalCLIDebugHook(ui_type="readline")]
        )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run()
