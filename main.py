import tensorflow as tf 
import pandas as pd
from sklearn.model_selection import train_test_split


def test_gpu():
    if tf.test.gpu_device_name():
        print('Default GPU Device {}'.format(tf.test.gpu_device_name()))
    else:
       print("Please install GPU version of TF")


def split_annotations_train_test(train_size=0.85):
    annotations_df = pd.read_csv('annotations.csv', names=['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'class'])
    print(annotations_df.head())
    print("Total boxes:", len(annotations_df))
    unique_names = annotations_df['image_path'].unique()
    train_names, test_names = train_test_split(unique_names, train_size=train_size, random_state=2020)
    print(train_names)
    train_annotations = annotations_df[annotations_df['image_path'].isin(train_names)]
    test_annotations = annotations_df[annotations_df['image_path'].isin(test_names)]
    train_annotations.to_csv('train_annotations.csv', index=False, header=None)
    test_annotations.to_csv('test_annotations.csv', index=False, header=None)
    print('Len train', len(train_annotations['image_path'].unique()))
    print('Len test', len(test_annotations['image_path'].unique()))


split_annotations_train_test()
