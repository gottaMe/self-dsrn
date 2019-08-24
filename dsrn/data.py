# coding=utf-8
import tensorflow as tf
import util


def dataset(hr_flist,
            lr_flist,
            scale,
            upsampling_method,
            num_epochs,
            resize=False,  # 这逼作者真是个gdx
            residual=True):
  """Build the TF sub graph for the inputdata pipeline."""
  with open(hr_flist, 'rb') as f:
    hr_filename_list = f.read().splitlines()
  with open(lr_flist, 'rb') as f:  # 作者给的代码有问题？？？？？。已改
    lr_filename_list = f.read().splitlines()
  with tf.name_scope('data_processiong'):
    filename_queue = tf.train.slice_input_producer(
        [hr_filename_list, lr_filename_list], num_epochs=num_epochs)
    hr_image_file = tf.read_file(filename_queue[0])
    hr_image = tf.image.decode_bmp(hr_image_file, channels=3)  # 此处解码得到的是uint8类型的tensor
    hr_image = tf.image.convert_image_dtype(hr_image, tf.float32)

    # target_scale = tf.random_uniform(shape=[1], minval=0.5, maxval=1.0)
    target_scale = tf.constant(1)
    hr_image = _rescale(hr_image, target_scale)
    lr_image = _rescale(hr_image, target_scale/scale)
    '''
    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      coord = tf.train.Coordinator()
      thread = tf.train.start_queue_runners(sess=sess, coord=coord)

      print(sess.run(target_scale))
      print(sess.run(hr_image).shape)
      print(sess.run(lr_image).shape)

      coord.request_stop()
      coord.join(thread)
    '''

    if (residual):
      hr_image = _make_residual(hr_image, lr_image, upsampling_method)
    hr_patches0, lr_patches0 = _make_patches(hr_image, lr_image, scale, resize,  # 此处不resize，得到的HR和LR的patches的个数会一样吗？
                                             upsampling_method)
    '''
    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      coord = tf.train.Coordinator()
      thread = tf.train.start_queue_runners(sess=sess, coord=coord)

      print(sess.run(hr_patches0).shape)
      print(sess.run(lr_patches0).shape)

      coord.request_stop()
      coord.join(thread)
    '''
    hr_patches1, lr_patches1 = _make_patches(
        tf.image.rot90(hr_image),  # 逆时针旋转90度
        tf.image.rot90(lr_image), scale, resize, upsampling_method)
    return tf.concat([hr_patches0, hr_patches1],
                     0), tf.concat([lr_patches0, lr_patches1], 0)


def _rescale(image, target_scale):  # 你TM乘法不会做？？？？？？
  with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)

    shape = sess.run(tf.shape(image)[:2])
    scale = sess.run(target_scale)
    new_shape = shape * scale  # tf.shape产生图像的[batch, height, width, channel]

    coord.request_stop()
    coord.join(thread)

    return tf.image.resize_images(image, tf.convert_to_tensor(new_shape, dtype=tf.int32), preserve_aspect_ratio=False)

def _make_residual(hr_image, lr_image, upsampling_method):
  """Compute the difference between HR and upsampled LR images"""
  hr_image = tf.expand_dims(hr_image, 0)
  lr_image = tf.expand_dims(lr_image, 0)
  hr_image_shape = tf.shape(hr_image)[1:3]
  res_image = hr_image - util.get_resize_func(upsampling_method)(lr_image,
                                                                 hr_image_shape)
  return tf.reshape(res_image, [hr_image_shape[0], hr_image_shape[1], 3])


def _make_patches(hr_image, lr_image, scale, resize, upsampling_method):
  """Extract patches from images, also apply augmentations"""
  hr_image = tf.stack(_flip([hr_image]))
  lr_image = tf.stack(_flip([lr_image]))
  hr_patches = util.image_to_patches(hr_image)
  if (resize):
    lr_image = util.get_resize_func(upsampling_method)(lr_image,
                                                       tf.shape(hr_image)[1:3])
    lr_patches = util.image_to_patches(lr_image)
  else:
    lr_patches = util.image_to_patches(lr_image, scale)
  return hr_patches, lr_patches


def _flip(img_list):
  flipped_list = []
  for img in img_list:
    flipped_list.append(
        tf.image.random_flip_up_down(
            tf.image.random_flip_left_right(img, seed=0), seed=0))
  return flipped_list
