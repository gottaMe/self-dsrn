# coding=utf-8
import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
import util
import os

flags = tf.app.flags
FLAGS = flags.FLAGS

# Task specification.
flags.DEFINE_string('hr_flist', 'train.txt',
                    'file_list containing the training data.')
flags.DEFINE_string('lr_flist', 'train.txt',
                    'file_list containing the lr_training data.')
flags.DEFINE_integer('scale', '2', 'batch size for training')

# Model and data preprocessing.
flags.DEFINE_string('data_name', 'data', 'Path to the data specification file.')
flags.DEFINE_string('model_name', 'model_recurrent_s2_u128_avg_t7', 'Path to the model specification file.')

flags.DEFINE_string('load_checkpoint', '',
                    'If given, load the checkpoint to initialize model.')
flags.DEFINE_string('output_dir', 'output_dir',
                    'Path to save the model checkpoint during training.')
flags.DEFINE_string('model_file_out', 'output_dir/model',
                    'Path to save the model')

# Training hyper parameters
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_integer('batch_size', 16, 'batch size.')
flags.DEFINE_float('ohnm', '1.0', 'percentage of hard negatives')
flags.DEFINE_integer('num_epochs', 1, 'number of epochs')
flags.DEFINE_string('upsampling_method', 'bicubic', 'nn or bicubic')

data = __import__(FLAGS.data_name)
model = __import__(FLAGS.model_name)


def build_data(g):
  """Build the data input pipeline."""
  with tf.device('/device:cpu:0'):
    with tf.name_scope('data'):
      target_patches, source_patches = data.dataset(  # 此处的target是HR，source是对应的LR
          FLAGS.hr_flist, FLAGS.lr_flist, FLAGS.scale, FLAGS.upsampling_method,
          FLAGS.num_epochs)
      target_batch_staging, source_batch_staging = tf.train.batch(
          [target_patches, source_patches],
          FLAGS.batch_size,
          # 32768,
          # 8192,
          num_threads=4,
          enqueue_many=True)
  '''
  with tf.name_scope('data_staging'):
    stager = data_flow_ops.StagingArea(
        [tf.float32, tf.float32],
        shapes=[[None, None, None, 3], [None, None, None, 3]])
    stage = stager.put([target_batch_staging, source_batch_staging])
    target_batch, source_batch = stager.get()
  '''

  return target_batch_staging, source_batch_staging  # 这里会不会少传了一个stage，在main中有个stage不晓得哪来的


def build_model(source, target):
  """Build the model graph."""
  with tf.name_scope('model'):
    prediction = model.build_model(
        source, FLAGS.scale, training=True, reuse=False)
    '''
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print(sess.run(source).shape)

        coord.request_stop()
        coord.join(threads)
    '''
    target_cropped = util.crop_center(target, tf.shape(prediction)[1:3])
    tf.summary.histogram('prediction', prediction)
    tf.summary.histogram('groundtruth', target)
  return prediction, target_cropped


# 特么你这函数竟然连return都没有...。已改
def build_loss(prediction, target_cropped):
  with tf.name_scope('l2_loss'):
    if FLAGS.ohnm < 1.0:
      pixel_loss = tf.reduce_sum(
          tf.square(tf.subtract(target_cropped, prediction)), 3)  # axis=3指的是通道数
      raw_loss = tf.reshape(pixel_loss, [-1])  # 把原来的tensor转变成一维tensor
      num_ele = tf.size(raw_loss)
      num_negative = tf.cast(
          tf.to_float(num_ele) * tf.constant(FLAGS.ohnm), tf.int32)
      hard_negative, _ = tf.nn.top_k(raw_loss, num_negative)  # 获得在所有的通道里面前ohnm个最大的loss
      avg_loss = tf.losses.mean_squared_error(target_cropped,
                                              prediction)

      hard_loss = tf.reduce_mean(hard_negative)
      tf.summary.scalar('training_l2_loss', avg_loss)
      tf.summary.scalar('training_hard_l2_loss', hard_loss)
      loss = hard_loss  # 此处的loss选择的是hard_l2_loss
    else:
      '''
      if FLAGS.precision > 0:
        loss = tf.reduce_mean(
            tf.square(
                tf.nn.relu(
                    tf.abs(target_cropped - prediction) -
                    FLAGS.precision / tf.uint8.max)))
      else:
      '''
      loss = tf.losses.mean_squared_error(target_cropped, prediction)

      tf.summary.scalar('training_l2_loss', loss)
  return loss


def build_trainer(loss):
  with tf.name_scope('train'):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    gvs = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in gvs]
    optimizer = optimizer.apply_gradients(capped_gvs, global_step=global_step)
    merged_summary_op = tf.summary.merge_all()
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

  return global_step, optimizer, merged_summary_op, init, saver


def prepare_directories(outdir):

  def make_dir(d):
    if not os.path.exists(d):
      os.mkdir(d)

  make_dir(outdir)
  ckpt_dir = os.path.join(outdir, "train")
  summary_dir = os.path.join(outdir, "summary")
  make_dir(ckpt_dir)
  make_dir(summary_dir)
  return ckpt_dir, summary_dir


def main(self):
  g = tf.Graph()
  with g.as_default():
    tgt, src = build_data(g)  # ？？？？？？？？？？？？？？前面是target后面才是source吧。已改
    pred, crop_tgt = build_model(src, tgt)  # 人函数返回两个值，一个预测值一个真值，还没算损失呢？？？？？。已改
    loss = build_loss(pred, crop_tgt)  # 此处使用的是l2_loss
    global_step, train_op, summary_op, init_op, saver = build_trainer(loss)  # 你前面可能忘记build_loss了，并且可能忘记回传saver了...

  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  config.gpu_options.per_process_gpu_memory_fraction = 0.95

  # Prepare output dir.
  ckpt_dir, summary_dir = prepare_directories(FLAGS.output_dir)

  with tf.Session(graph=g, config=config) as sess:
    with tf.device('/device:cpu:0'):
        sess.run(init_op)
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        train_writer = tf.summary.FileWriter(summary_dir, sess.graph)

        if tf.gfile.Exists(FLAGS.load_checkpoint):
          print('Loading model from %s', FLAGS.load_checkpoint)
          saver.restore(sess, FLAGS.load_checkpoint)  # 看来是上面build_train的时候没把saver传回来...

        try:
          # sess.run(stage)  # 这里的stage是哪来的？并且为什么要再run一次?
          step = 0
          while not coord.should_stop():
            _, training_loss, train_summary = sess.run([
                # stage,
                train_op,
                # step,  # step又是哪里来的...
                loss,
                summary_op,
            ])
            '''
            # print(0)
            # sess.run(stage)
            print(1)
            training_loss = sess.run(loss)
            print(2)
            sess.run(train_op)
            print(3)
            train_summary = sess.run(summary_op)
            '''
            print('Training at step %d, loss=%f' % (step, training_loss))
            train_writer.add_summary(train_summary, step)
            if (step % 1000 == 0):
              saver.save(sess, ckpt_dir, global_step=global_step)
            step += 1
        except tf.errors.OutOfRangeError:
          print('Done training -- epoch limit reached')
        finally:
          coord.request_stop()
          coord.join(threads)
          saver.save(sess, FLAGS.model_file_out, global_step=global_step)


if __name__ == '__main__':
  tf.app.run()
