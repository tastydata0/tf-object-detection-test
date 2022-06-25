import tensorflow.compat.v1 as tf
import argparse
import os
import io
from PIL import Image
from object_detection.utils import dataset_util


# Initiate argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-a",
                    "--annots",
                    type=str)
parser.add_argument("-i",
                    "--images_path",
                    type=str)
parser.add_argument("-o",
                    "--output_path",
                    help="Path of output TFRecord (.record) file.", type=str)
args = parser.parse_args()

def create_tf_example(example):
  # TODO(user): Populate the following variables from your example.
  filename = example['filename'] # Filename of the image. Empty if image is not from file
  encoded_image_data = None # Encoded image bytes
  with tf.gfile.GFile(os.path.join(args.images_path, '{}'.format(filename)), 'rb') as fid:
        encoded_image_data = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_image_data)
  image = Image.open(encoded_jpg_io)
  width, height = image.size
  
  
  image_format = b'jpeg' # b'jpeg' or b'png'

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  # classes_text = [] # List of string cass name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)

  for i in range(len(example['data'])//4):
    x = float(example['data'][i*4])
    y = float(example['data'][i*4+1])
    w = float(example['data'][i*4+2])
    h = float(example['data'][i*4+3])
    xmins.append(x/width)
    xmaxs.append((x+w)/width)
    ymins.append(y/height)
    ymaxs.append((y+h)/height)
    classes_text.append(b'person')
    classes.append(1)


  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example


def main(_):
  writer = tf.python_io.TFRecordWriter(args.output_path)

  # TODO(user): Write code to read in your dataset to examples variable
  with open(args.annots) as annots:
    for line in annots:
      example = {}

      splitted = line.split(' ')
      if len(splitted) == 1:
        continue
      example['filename'] = splitted[0]
      example['data'] = splitted[1:]

      tf_example = create_tf_example(example)
      writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.app.run()