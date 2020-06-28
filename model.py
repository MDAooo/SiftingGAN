from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *
from data_prep import *  # 自定义数据加载


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class SiftingGAN(object):
    def __init__(self, sess, input_height=256, input_width=256, crop=True,
                 batch_size=64, sample_num=64, output_height=64, output_width=64,
                 y_dim=30, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):
        """

        Args:
          sess: TensorFlow session
          crop: "True for training, False for testing [False]"
          batch_size: The size of batch. Should be specified before training.
          y_dim: (optional) Dimension of dim for y. [None]
          z_dim: (optional) Dimension of dim for Z. [100]
          gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
          df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
          dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.crop = crop

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.d_bn4 = batch_norm(name='d_bn4')
        self.d_bn5 = batch_norm(name='d_bn5')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        self.g_bn3 = batch_norm(name='g_bn3')
        self.g_bn4 = batch_norm(name='g_bn4')
        self.g_bn5 = batch_norm(name='g_bn5')
        self.g_bny0 = batch_norm(name='g_bny0')

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.c_dim = 3
        # self.data_X, self.data_y = get_suffled_datalist(self.dataset_name, self.input_fname_pattern)
        # imreadImg = imread(self.data_X[0])
        # if len(imreadImg.shape) >= 3:  # check if image is a non-grayscale image by checking channel number
        #     self.c_dim = imreadImg.shape[-1]
        # else:
        #     self.c_dim = 1

        self.build_model()

    def build_model(self):
        if self.y_dim:
            self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')  # [64,10]

        if self.crop:
            image_dims = [self.output_height, self.output_width, self.c_dim]  # training
        else:
            image_dims = [self.input_height, self.input_width, self.c_dim]  # testing

        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims,
            name='real_images')  # training:[64,64,64,3]crop==True 中心裁剪跟输出大小一致

        inputs = self.inputs

        self.z = tf.placeholder(
            tf.float32, [None, self.z_dim], name='z')
        self.z_sum = histogram_summary("z", self.z)

        self.G = self.generator(self.z, self.y)
        self.D, self.D_logits = self.discriminator(inputs, self.y, reuse=False)
        self.sampler = self.sampler(self.z, self.y)
        self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)

        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_)
        self.G_sum = image_summary("G", self.G)

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        self.d_loss_real = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.g_sum = merge_summary([self.z_sum, self.d__sum,
                                    self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary(
            [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = SummaryWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))  # [-1,1]之间均匀分布，size=[64,100]
        sample_image = cv2.imread('/data/sample_image.png')
        sample_inputs = np.array(self.trim_image(sample_image, 8, 8)).astype(np.float32)
        y_label = [i for i in range(self.y_dim)]
        y_label = (y_label + y_label + y_label)[0:64]
        sample_labels = self.sess.run(tf.one_hot(y_label, self.y_dim, 1, 0))
        # display labeled samples in class order. 按顺序显示每一个类别的sample图像
        # sample_files, label_batch = get_sample_data("AID", "*.jpg", self.sample_num, self.y_dim)
        # sample_labels = self.sess.run(tf.one_hot(label_batch, self.y_dim, 1, 0))
        # sample = [
        #     get_image(sample_file,
        #               input_height=self.input_height,
        #               input_width=self.input_width,
        #               resize_height=self.output_height,
        #               resize_width=self.output_width,
        #               crop=self.crop) for sample_file in sample_files]
        # sample_inputs = np.array(sample).astype(np.float32)
        # save_images(sample_inputs, image_manifold_size(sample_inputs.shape[0]),
        #             './sample_image.png')

        # 从tfrecord中随机读取sample
        # sample_inputs, _, sample_labels = self.sess.run([img_batch, label_batch, y_batch])

        # sample_files = self.data_X[0:self.sample_num]
        # label_batch = self.data_y[0:self.sample_num]
        # sample_labels = self.sess.run(tf.one_hot(label_batch, self.y_dim, 1, 0))
        # sample = [
        #   get_image(sample_file,
        #             input_height=self.input_height,
        #             input_width=self.input_width,
        #             resize_height=self.output_height,
        #             resize_width=self.output_width,
        #             crop=self.crop,
        #             grayscale=self.grayscale) for sample_file in sample_files]
        # if (self.grayscale):
        #   sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
        # else:
        #   sample_inputs = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # read data from tfrecord.  从tfrecord中读取数据,不用tfrecord训练一个batch约2+1s
        img, label = tfrecord_read()
        img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                        batch_size=self.batch_size,
                                                        capacity=4000,
                                                        min_after_dequeue=2000)
        y_batch = tf.one_hot(label_batch, self.y_dim, 1, 0)
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        for epoch in xrange(config.epoch):

            batch_idxs = config.train_size // config.batch_size

            for idx in xrange(0, batch_idxs):
                batch_images, _, batch_labels = self.sess.run([img_batch, label_batch, y_batch])

                # a batch of random noise z
                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                    .astype(np.float32)

                ### training
                # Update D network

                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={
                                                   self.inputs: batch_images,
                                                   self.z: batch_z,
                                                   self.y: batch_labels,
                                               })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={
                                                   self.z: batch_z,
                                                   self.y: batch_labels,
                                               })
                self.writer.add_summary(summary_str, counter)

                # double g_optim to make sure that d_loss does not go to zero
                if counter > 25000:
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                   feed_dict={self.z: batch_z, self.y: batch_labels})
                    self.writer.add_summary(summary_str, counter)
                elif counter > 50000:
                    for _ in range(3):
                        _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                       feed_dict={self.z: batch_z, self.y: batch_labels})
                        self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({
                    self.z: batch_z,
                    self.y: batch_labels
                })
                errD_real = self.d_loss_real.eval({
                    self.inputs: batch_images,
                    self.y: batch_labels
                })
                errG = self.g_loss.eval({
                    self.z: batch_z,
                    self.y: batch_labels
                })

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, batch_idxs,
                         time.time() - start_time, errD_fake + errD_real, errG))
                if counter < 30000:
                    if np.mod(counter, 10) == 1:
                        try:
                            samples, d_loss, g_loss = self.sess.run(
                                [self.sampler, self.d_loss, self.g_loss],
                                feed_dict={
                                    self.z: sample_z,
                                    self.inputs: sample_inputs,
                                    self.y: sample_labels,
                                }
                            )
                            save_images(samples, image_manifold_size(samples.shape[0]),
                                        './{}/train_{:06d}.png'.format(config.sample_dir, counter))
                            print("[Sample] counter:%.6d  d_loss: %.8f, g_loss: %.8f" % (counter, d_loss, g_loss))
                        except:
                            print("display pic error!...")
                else:
                    try:
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={
                                self.z: sample_z,
                                self.inputs: sample_inputs,
                                self.y: sample_labels,
                            }
                        )
                        #  Generative-Model-Sifting
                        if g_loss < config.GMS_thres:
                            save_images(samples, image_manifold_size(samples.shape[0]),
                                        './{}/train_{:06d}.png'.format(config.sample_dir, counter))
                            print("[Sample] counter:%.6d  d_loss: %.8f, g_loss: %.8f" % (counter, d_loss, g_loss))
                    except:
                        print(" Generative-Model-Sifting output error!...")

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

        coord.request_stop()
        coord.join(threads)

    def discriminator(self, image, y=None, reuse=False):
        mfn = 16
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

                # 'UCMerced_LandUse' dataset y_dim=21
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            yb_ = yb * tf.ones([self.batch_size, self.input_height, self.input_width, self.y_dim])

            h0 = lrelu(conv2d(image, mfn, name='d_h0_conv'))  # conv2d [64,256,256,3+21]->[64,128,128,16+21] +lrelu
            y0 = lrelu(conv2d(yb_, mfn, name='d_y0_conv'))
            h0 = concat([h0, y0], 3)

            h1 = lrelu(
                self.d_bn1(
                    conv2d(h0, 2 * mfn, name='d_h1_conv')))  # conv2d [64,128,128,37+21]->[64,64,64,32+21] bn+lrelu

            h2 = lrelu(self.d_bn2(conv2d(h1, 4 * mfn, name='d_h2_conv')))  # conv2d [64,64,64,74]->[64,32,32,64+21]

            h3 = lrelu(self.d_bn3(conv2d(h2, 8 * mfn, name='d_h3_conv')))  # conv2d [64,32,32,106]->[64,16,16,128+21]

            h4 = lrelu(self.d_bn4(conv2d(h3, 16 * mfn, name='d_h4_conv')))  # conv2d [64,16,16,149+21]->[64,8,8,256+21]

            h5 = lrelu(self.d_bn5(conv2d(h4, 32 * mfn, name='d_h5_conv')))  # conv2d [64,8,8,277+21]->[64,4,4,512+21]
            h5 = tf.reshape(h5, [self.batch_size, -1])  # [64,4*4*533]

            h6 = linear(h5, 1, 'd_h6_lin')  # linear [64,4*4*533+21]->[64,1]

            return tf.nn.sigmoid(h6), h6  # sigmiod

    def generator(self, z, y=None):
        mfn = 16
        with tf.variable_scope("generator") as scope:
            s_h, s_w = self.output_height, self.output_width  # 256,256
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)  # 128
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)  # 64
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)  # 32
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)  # 16
            s_h32, s_w32 = conv_out_size_same(s_h16, 2), conv_out_size_same(s_w16, 2)  # 8
            s_h64, s_w64 = conv_out_size_same(s_h32, 2), conv_out_size_same(s_w32, 2)  # 4

            yb = tf.cast(tf.reshape(y, [self.batch_size, self.y_dim]), tf.float32)  # [64,21]

            self.z_, self.h0_w, self.h0_b = linear(
                z, 16 * mfn * s_h64 * s_h64, 'g_h0_lin', with_w=True)  # linear z[64,100]->[64,512*4*4]
            self.h0 = tf.reshape(self.z_, [-1, s_h64, s_h64, 16 * mfn])  # -1 ->dimension/(s_h16*s_w16*self.gf_dim * 8) = 64
            h0 = tf.nn.relu(self.g_bn0(self.h0))  # [64,4,4,512] relu

            y0_, self.h0_w, self.h0_b = linear(
                yb, 16 * mfn * s_h64 * s_h64, 'g_y0_lin', with_w=True)  # linear z[64,21]->[64,512*4*4]
            y0 = tf.reshape(y0_, [-1, s_h64, s_h64, 16 * mfn])
            y0 = tf.nn.relu(self.g_bny0(y0))
            h0 = concat([h0, y0], 3)  # [64,4,4,1024]

            self.h1, self.h1_w, self.h1_b = deconv2d(
                h0, [self.batch_size, s_h32, s_w32, 16 * mfn], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))  # deconv2d [64,4,4,533]->[64,8,8,256] bn+relu

            h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, s_h16, s_w16, 8 * mfn], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))  # deconv2d ->[64,16,16,128] bn+relu

            h3, self.h3_w, self.h3_b = deconv2d(h2, [self.batch_size, s_h8, s_w8, 4 * mfn], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))  # deconv2d ->[64,32,32,64] bn+relu

            h4, self.h4_w, self.h4_b = deconv2d(h3, [self.batch_size, s_h4, s_w4, 2 * mfn], name='g_h4', with_w=True)
            h4 = tf.nn.relu(self.g_bn4(h4))  # deconv2d ->[64,64,64,32] bn+relu

            h5, self.h5_w, self.h5_b = deconv2d(h4, [self.batch_size, s_h2, s_w2, 2 * mfn], name='g_h5', with_w=True)
            h5 = tf.nn.relu(self.g_bn5(h5))  # deconv2d ->[64,128,128,16] bn+relu

            h6, self.h6_w, self.h6_b = deconv2d(h5, [self.batch_size, s_h, s_w, self.c_dim], name='g_h6', with_w=True)
            # deconv2d ->[64,256,256,3] tanh ->return

            return tf.nn.tanh(h6)

    def sampler(self, z, y=None):
        mfn = 16
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s_h, s_w = self.output_height, self.output_width  # 256,256
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)  # 128
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)  # 64
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)  # 32
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)  # 16
            s_h32, s_w32 = conv_out_size_same(s_h16, 2), conv_out_size_same(s_w16, 2)  # 8
            s_h64, s_w64 = conv_out_size_same(s_h32, 2), conv_out_size_same(s_w32, 2)  # 4

            yb = tf.reshape(y, [self.batch_size, self.y_dim])  # [64,21]

            self.z_, self.h0_w, self.h0_b = linear(
                z, 16 * mfn * s_h64 * s_h64, 'g_h0_lin', with_w=True)  # linear z[64,100]->[64,512*4*4]
            self.h0 = tf.reshape(self.z_, [-1, s_h64, s_h64, 16 * mfn])  # -1 ->dimension/(s_h16*s_w16*self.gf_dim * 8) = 64
            h0 = tf.nn.relu(self.g_bn0(self.h0))  # [64,4,4,512] relu

            y0_, self.h0_w, self.h0_b = linear(
                yb, 16 * mfn * s_h64 * s_h64, 'g_y0_lin', with_w=True)  # linear z[64,21]->[64,512*4*4]
            y0 = tf.reshape(y0_, [-1, s_h64, s_h64, 16 * mfn])
            y0 = tf.nn.relu(self.g_bny0(y0))
            h0 = concat([h0, y0], 3)  # [64,4,4,1024]

            self.h1, self.h1_w, self.h1_b = deconv2d(
                h0, [self.batch_size, s_h32, s_w32, 16 * mfn], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))  # deconv2d [64,4,4,533]->[64,8,8,256] bn+relu

            h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, s_h16, s_w16, 8 * mfn], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))  # deconv2d ->[64,16,16,128] bn+relu

            h3, self.h3_w, self.h3_b = deconv2d(h2, [self.batch_size, s_h8, s_w8, 4 * mfn], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))  # deconv2d ->[64,32,32,64] bn+relu

            h4, self.h4_w, self.h4_b = deconv2d(h3, [self.batch_size, s_h4, s_w4, 2 * mfn], name='g_h4', with_w=True)
            h4 = tf.nn.relu(self.g_bn4(h4))  # deconv2d ->[64,64,64,32] bn+relu

            h5, self.h5_w, self.h5_b = deconv2d(h4, [self.batch_size, s_h2, s_w2, 2 * mfn], name='g_h5', with_w=True)
            h5 = tf.nn.relu(self.g_bn5(h5))  # deconv2d ->[64,128,128,16] bn+relu

            h6, self.h6_w, self.h6_b = deconv2d(h5, [self.batch_size, s_h, s_w, self.c_dim], name='g_h6', with_w=True)
            # deconv2d ->[64,256,256,3] tanh ->return

            return tf.nn.tanh(h6)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "siftingGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    # Labeled-Sample-Discriminating 利用判别器对生成的样本进行筛选
    def LSD(self, sample_dir, sifted_dir, threshold):

        if not os.path.exists(sifted_dir):
            os.makedirs(sifted_dir)
        # classlist = os.listdir("D://1Experiment\AID_dataprocess\data\AID")
        with open('/data/AID_ClassList','r') as f:
            classlist = f.read().splitlines()
            for c in classlist:
                if not os.path.exists(os.path.join(sifted_dir, c)):
                    os.makedirs(os.path.join(sifted_dir, c))

        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        slt_class_list = os.listdir(sifted_dir)
        file_list = os.listdir(sample_dir)
        for file_name in file_list:
            image = cv2.imread(os.path.join(sample_dir, file_name))
            img_list = self.trim_image(image, 8, 8)
            img_batch = np.array(img_list, np.float32)/127.5 - 1

            y_label = [i for i in range(self.y_dim)]
            y_label = (y_label+y_label+y_label)[0:64]
            y_batch = self.sess.run(tf.one_hot(y_label, self.y_dim, 1, 0))
            D_prob= self.sess.run([self.D],
                                           feed_dict={
                                               self.inputs: img_batch,
                                               self.y: y_batch,
                                           })
            for idx in range(self.batch_size):
                if(D_prob[0][idx,0] > threshold):
                    print(file_name,idx)
                    self.save_image(img_list[idx], os.path.join(sifted_dir,slt_class_list[idx % self.y_dim]), file_name[0:-4]+"-"+str(idx)+file_name[-4:])

    def save_image(self, image, path, img_name):
        scipy.misc.imsave(os.path.join(path, img_name), image)

    def trim_image(self, image, r, c):
        imlist = []
        h, w, t = np.shape(image)
        trim_h, trim_w = round(h / r), round(w / c)
        for i in range(0, r):
            for j in range(0, c):
                im = image[i * trim_h:(i + 1) * trim_h, j * trim_w:(j + 1) * trim_w]
                imlist.append(im)
        return imlist