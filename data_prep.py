import os
from glob import glob
import tensorflow as tf
import scipy.misc
import numpy as np
import cv2

data_path="D:/1Experiment/AID_dataprocess/data"

def tfrecord_write_suffled(record_name="train.tfrecords", dataset_name='UCMerced_LandUse', input_fname_pattern='*.tif'):
    data_list, label_list = get_suffled_datalist(dataset_name, input_fname_pattern)
    print(len(data_list))
    print(label_list)
    writer = tf.python_io.TFRecordWriter(record_name)
    for index, img_path in enumerate(data_list):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256))
        img_raw = img.tobytes();
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label_list[index]])),
            "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())
    writer.close()


def tfrecord_write(record_name="train.tfrecords", dataset_name='UCMerced_LandUse', input_fname_pattern='*.tif'):
    writer = tf.python_io.TFRecordWriter(record_name)
    counter = 0
    class_list = os.listdir(os.path.join("./data", dataset_name))
    for index, class_dir in enumerate(class_list):
        file_names = glob(os.path.join("./data", dataset_name, class_dir, input_fname_pattern))
        for img_path in file_names:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (256, 256))
            img_list = image_transform(img)
            write_image(img_list, img_path)
            for im in img_list:
                img_raw = im.tobytes();
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
                writer.write(example.SerializeToString())
                counter += 1
            print(counter)
    writer.close()


def image_transform(image):
    image_f0 = cv2.flip(image, 0)  # 上下翻转
    image_f1 = cv2.flip(image, 1)  # 左右翻转
    image_r90 = cv2.rotate(image, 0)
    image_r180 = cv2.rotate(image, 1)
    image_r270 = cv2.rotate(image, 2)
    image_f0r90 = cv2.rotate(image_f0, 0)
    image_f0r270 = cv2.rotate(image_f0, 2)
    image_list = [image, image_f0, image_f1, image_r90, image_r180, image_r270, image_f0r90, image_f0r270]
    return image_list


def write_image(image_list, path):
    for i in range(1, len(image_list)):
        image_name = path[0:len(path) - 4] + "_%s.tif" % (i)
        cv2.imwrite(image_name, image_list[i])


def tfrecord_read(record_name="/data/train233.tfrecords"):
    filename_queue = tf.train.string_input_producer([record_name])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string)
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [256, 256, 3])
    img = tf.cast(img, tf.float32) / 127.5 - 1
    label = tf.cast(features['label'], tf.int32)

    return img, label


'''
直接这样用会造成线程堵塞，需要使用Coordinator类用来帮助多个线程协同工作，多个线程同步终止。
其主要方法有：
should_stop():如果线程应该停止则返回True。
request_stop(<exception>): 请求该线程停止。
join(<list of threads>):等待被指定的线程终止。
首先创建一个Coordinator对象，然后建立一些使用Coordinator对象的线程。
这些线程通常一直循环运行，一直到should_stop()返回True时停止。 
任何线程都可以决定计算什么时候应该停止。它只需要调用request_stop()，同时其他线程的should_stop()将会返回True，然后都停下来。
with tf.Session() as sess:  
    # Start populating the filename queue.  
    # Coordinator类用来帮助多个线程协同工作，多个线程同步终止  
    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(coord=coord)  
  
    train_steps = 10  
    # Retrieve a single instance:  
    try:  
        while not coord.should_stop():  # 如果线程应该停止则返回True  
            example, label = sess.run([example_batch, label_batch])  
            print (example)  
  
            train_steps -= 1  
            print train_steps  
            if train_steps <= 0:  
                coord.request_stop()    # 请求该线程停止  
  
    except tf.errors.OutOfRangeError:  
        print ('Done training -- epoch limit reached')  
    finally:  
        # When done, ask the threads to stop. 请求该线程停止  
        coord.request_stop()  
        # And wait for them to actually do it. 等待被指定的线程终止  
        coord.join(threads)  
'''


def train_test():
    img, label = tfrecord_read()
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=64,
                                                    capacity=4000,
                                                    min_after_dequeue=2000)
    y = tf.one_hot(label_batch, 21, on_value=0.5, off_value=0., dtype=tf.float32)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        threads = tf.train.start_queue_runners(sess)
        for i in range(2):
            val, l, b = sess.run([img_batch, label_batch, y])
            # val, l = sess.run([img, label])
            print(val.shape, l)
            print(b, y.shape)


def get_data_list(dataset_name, input_fname_pattern):
    data_list = []
    label_list = []
    class_list = os.listdir(os.path.join(data_path, dataset_name))
    for index, class_dir in enumerate(class_list):
        file_names = glob(os.path.join(data_path, dataset_name, class_dir, input_fname_pattern))
        data_list += file_names
        label_list += [index for i in range(len(file_names))]
    return data_list, label_list


def get_suffled_datalist(dataset_name, input_fname_pattern):
    data_list, label_list = get_data_list(dataset_name, input_fname_pattern)
    seed = 233
    np.random.seed(seed)
    np.random.shuffle(data_list)
    np.random.seed(seed)
    np.random.shuffle(label_list)
    return data_list, label_list


def get_sample_data(dataset_name, input_fname_pattern, sample_num, y_dim):
    num = 200
    idxs = np.random.randint(0, num, int(np.ceil(sample_num/y_dim)))
    sample_files = []
    sample_lables = []

    class_list = os.listdir(os.path.join(data_path, dataset_name))
    for index, class_dir in enumerate(class_list):
        file_names = glob(os.path.join(data_path, dataset_name, class_dir, input_fname_pattern))
        sample_files.append(file_names[0])
        sample_lables.append(index)
    sample_files += sample_files
    sample_lables += sample_lables
    sample_files += sample_files
    sample_lables += sample_lables
    return sample_files[0:sample_num], sample_lables[0:sample_num]


def select(sample_dir, select_dir):
    if not os.path.exists(select_dir):
        os.makedirs(select_dir)
    classlist = os.listdir("D://1Experiment\AID_dataprocess\data\AID")
    for c in classlist:
        if not os.path.exists(os.path.join(select_dir, c)):
            os.makedirs(os.path.join(select_dir, c))
    file_list = os.listdir(sample_dir)
    for file_name in file_list:
        image = cv2.imread(os.path.join(sample_dir,file_name))
        img_list = trim_image(image, file_name, 8, 8)
        print(np.shape(img_list))

def trim_image(image,file_name,r,c):
    imlist = []
    h,w,t=np.shape(image)
    trim_h,trim_w=round(h/r),round(w/c)
    for i in range(0,r):
        for j in range(0,c):
            im=image[i*trim_h:(i+1)*trim_h,j*trim_w:(j+1)*trim_w]
            imlist.append(im)
    return imlist
# if __name__ == '__main__':
    # sample_files, label_batch = get_sample_data("AID", "*.jpg", 64, 30)
    # print(sample_files, label_batch)
    # tfrecord_write()
    # data_list, label_list = get_suffled_datalist("UCMerced_LandUse", "*.tif")
    # print(data_list[0:3])
    # print(label_list)
    # tfrecord_write_suffled()
    # y = 0
    # yi_batch = [0 for _ in range(64)]
    # print(yi_batch)
    # y_label = [i for i in range(30)]
    # y_label = (y_label + y_label + y_label)[0:64]
    # print(y_label)
