import tensorflow as tf
import random
import pathlib
# 下载测试数据
# data_root_orig = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
#                                          fname='flower_photos', untar=True)
# data_root = pathlib.Path(data_root_orig)
# print(data_root)  # 打印出数据集所在目录

def load_and_preprocess_from_path_label(path, label):
    image = tf.io.read_file(path)  # 读取图片
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])  # 原始图片大小为(266, 320, 3)，重设为(192, 192)
    image /= 255.0  # 归一化到[0,1]范围
    return image, label

data_path = pathlib.Path("../../data/flower_photos")
all_image_paths = list(data_path.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)

label_names = sorted(item.name for item in data_path.glob('*/') if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))

all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))

ds = ds.map(load_and_preprocess_from_path_label)
ds = ds.repeat(1)
ds = ds.shuffle(buffer_size=image_count)
ds = ds.batch(32)
iterator = iter(ds)

while 1:
    try:
        image, label = next(iterator)
        print(image.shape)
        print(label.shape)
    except StopIteration:   # python内置的迭代器越界错误类型
        print("iterator done")
        break
