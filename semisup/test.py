import tensorflow as tf
from time import time
import numpy as np
import tensorflow.contrib.eager as tfe
#tfe.enable_eager_execution()
#asd = tfe.Variable(1)
#n = #files = tf.data.Dataset.list_files(n)
#t = files.make_one_shot_iterator()

#dataset = tf.data.TFRecordDataset(files,num_parallel_reads=1)

#dataset = dataset.batch(1)
#dataset.repeat()
#iterator = dataset.make_one_shot_iterator()

#next_element = t.get_next()

def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

# A vector of filenames.
filenames = tf.gfile.ListDirectory("/data/Dropbox/Data_new/id_workplace/lviv_data/idscam-dataset/test/blaupunkt/")
filenames = ["/data/Dropbox/Data_new/id_workplace/lviv_data/idscam-dataset/test/blaupunkt/"+f for f in filenames]

# `labels[i]` is the label for the image in `filenames[i].
labels = tf.constant([1 for _ in range(len(filenames))])
print labels
'''dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.shuffle(len(filenames))
dataset = dataset.map(_parse_function,num_parallel_calls=16).repeat()
dataset = dataset.batch(100)
dataset = dataset.apply(tf.contrib.data.prefetch_to_device('/gpu:0'))
iterator = dataset.make_one_shot_iterator()
im_ba = iterator.get_next()
start = time()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        imgs, lbls  = sess.run(im_ba)
    print len(imgs)
    print tf.one_hot(5,6).eval()
#for i in range(10):
#        imgs, lbls  = iterator.get_next()

print time()-start
print (imgs.shape)
'''
def make_datasets(n):
    dataset = tf.data.Dataset.range(n).repeat().batch(n)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    return next_element
a=[]
for i in range (10):
    a.append(make_datasets(i+1))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(200):
        for j in range(10):
            value = sess.run(a[j])
            print (value)