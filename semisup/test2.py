from __future__ import print_function
import tensorflow as tf
print (tf.__version__)
from semisup.tools import imagenetLV
import numpy as np
import cv2
import matplotlib

#
import seaborn.apionly as sns
import matplotlib.pyplot as plt
#matplotlib.use('TKAgg')
print (matplotlib.get_backend())
import tfplot

#plt.interactive(True)
#plt.figure(1)
#sns.set()
#points = np.arange(-5, 5, 0.01)
#dx, dy = np.meshgrid(points, points)
#z = (np.sin(dx)+np.sin(dy))
#plt.imshow(z)
#plt.colorbar()
#plt.title('plot for sin(x)+sin(y)')
#plt.show()
#tf_heatmap = tfplot.wrap_axesplot(sns.heatmap, annot=True, fmt="d", batch=True)
#tf.summary.image("attention_maps", tf_heatmap(attention_maps))

input_shape = (224,224)
con_mat = tf.confusion_matrix(labels=[0, 1, 2, 3, 4, 5, 6,1], predictions=[0,1,2,3,4,5,6,6], dtype=tf.int32, name=None)
def cp(a):
    print((a))
    return a
init = tf.initialize_all_variables()



with tf.Session():
    print (con_mat.eval())
    #sns.heatmap(con_mat.eval(),annot=True)
    tf_heatmap = tfplot.wrap_axesplot(sns.heatmap, figsize=(4, 4), tight_layout=True,
                                      cmap='jet', cbar=False, xticklabels=False, yticklabels=False)
    tf_heatmap(con_mat.eval()).eval()
    #plt.show()
    #plt.pause(10)
    rng = np.random.RandomState()
    for i in range(10):print (tf.random_uniform([], 0.0, 1).eval())
    def _random_invert(inputs1):

        inputs = tf.cast(inputs1, tf.float32)
        inputs = tf.image.adjust_brightness(inputs, tf.random_uniform((1,1),0.0,0.5))
        inputs = tf.image.random_contrast(inputs, 0.3, 1)
        #inputs = tf.image.per_image_standardization(inputs)
        inputs = tf.image.random_hue(inputs, 0.05)
        inputs = tf.image.random_saturation(inputs, 0.5, 1.1)
        '''
        inputs = tf.cast(inputs, tf.float32)
        inputs = tf.image.random_brightness(inputs, 1)
        inputs = tf.image.random_contrast(inputs, 0, 0.1, seed=10)
        inputs = tf.image.random_hue(inputs, 0.2, seed=10)
        inputs = tf.image.random_saturation(inputs, 0, 1, seed=10)
        '''

        def f1(): return tf.abs(inputs)

        def f2(): return tf.abs(inputs1)

        cond = tf.less(tf.random_uniform([], 0.0, 1), 0.5)
        return tf.cond(cond, f1, f2)


    images,labels = imagenetLV.get_data('train')
    print (images[0])
    for i in images:
        break
        image = tf.read_file(np.array(i))
        image_decoded = tf.cast(tf.image.decode_jpeg(image, channels=3), tf.float32)
        image_resized = tf.image.resize_images(image_decoded, (input_shape[0], input_shape[1])).eval()/255.0
        image_resized = cv2.cvtColor(image_resized,cv2.COLOR_RGB2BGR)
        img_aug = _random_invert(image_resized).eval()
        img_aug = np.hstack((image_resized,img_aug))
        cv2.imshow('test',img_aug)
        if chr(cv2.waitKey()) == 'q': break
    cv2.destroyAllWindows()
    #x = tf.Print(con_mat,[cp(con_mat.eval())],message='test:',summarize=89)
    #tf.Tensor.eval(x)
    #print('Confusion Matrix: \n\n', tf.Tensor.eval(con_mat,feed_dict=None, session=None))