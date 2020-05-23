import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image, ImageDraw
from ast import literal_eval
import matplotlib.pyplot as plt
import urllib
from tqdm import tqdm
import tensorflow as tf
import os


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# os.system('cd keras-retinanet')
# os.system('pip3 install .')
# os.system('python3 setup.py build_ext --inplace')
# os.system('cd ../')

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

root = 'dataset/'
train_img = root + "train"
test_img = root + "test"
train_csv = root + "train.csv"
sample_submission = root + "sample_submission.csv"

train = pd.read_csv(train_csv)
train.head()

print('Total images:', len(train['image_id'].unique()))
print("Total Bboxes", train.shape[0])
print(train['width'].unique())
print(train['height'].unique())


def get_bbox_area(bbox):
    bbox = literal_eval(bbox)
    return bbox[2] * bbox[3]


train['bbox_area'] = train['bbox'].apply(get_bbox_area)
train['bbox_area'].value_counts().hist(bins=10)

unique_images = train['image_id'].unique()
len(unique_images)

# In[14]:


num_total = len(os.listdir(train_img))
num_annotated = len(unique_images)

print(
    "There are {} annotated images and {} images without annotations.".format(num_annotated, num_total - num_annotated))

sources = train['source'].unique()
print("There are {} sources of data: {}".format(len(sources), sources))

train['source'].value_counts()


# plt.hist(train['image_id'].value_counts(), bins=10)
# plt.show()


def show_images(images, num=5):
    images_to_show = np.random.choice(images, num)

    for image_id in images_to_show:

        image_path = os.path.join(train_img, image_id + ".jpg")
        image = Image.open(image_path)

        # get all bboxes for given image in [xmin, ymin, width, height]
        bboxes = [literal_eval(box) for box in train[train['image_id'] == image_id]['bbox']]

        # visualize them
        draw = ImageDraw.Draw(image)
        for bbox in bboxes:
            draw.rectangle([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], width=3)

        plt.figure(figsize=(15, 15))
        plt.imshow(image)
        plt.show()


# show_images(unique_images)

bboxs = [bbox[1:-1].split(', ') for bbox in train['bbox']]
bboxs = [
    '{},{},{},{},wheat'.format(int(float(bbox[0])),
                         int(float(bbox[1])),
                         int(float(bbox[0])) + int(float(bbox[2])),
                         int(float(bbox[1])) + int(float(bbox[3])))

    for bbox in bboxs]
train['bbox_'] = bboxs
train.head()

# In[21]:


train_df = train[['image_id', 'bbox_']]
train_df.head()

# In[22]:


train_df = train_df.sample(frac=1).reset_index(drop=True)
train_df.head()

# ## Preparing Files to be given for training
# 
# ### Annotation file contains all the path of all images and their corresponding bounding boxes
# ### Class file contains the number of classes but in our case it is just 1 (Wheat)

# In[23]:


with open("annotations.csv", "w") as file:
    for idx in range(len(train_df)):
        file.write(train_img + "/" + train_df.iloc[idx, 0] + ".jpg" + "," + train_df.iloc[idx, 1] + "\n")

# In[24]:


with open("classes.csv", "w") as file:
    file.write("wheat,0")

# In[25]:


# ! wget 'https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5'


# ## Downloading the pretrained model

# ### Model Parameters

# In[ ]:


PRETRAINED_MODEL = 'resnet50_csv_02.h5'
EPOCHS = 1
BATCH_SIZE = 4
STEPS = len(train_df)//BATCH_SIZE + 1 #Keeping it small for faster commit
LR = 1e-4

print('TRAIN')
os.system('python3 keras-retinanet/keras_retinanet/bin/train.py --random-transform --weights {} --lr '
          '{} --batch-size {} --steps {} --epochs {} --image-min-side 512 --image-max-side 512 '
          'csv annotations.csv classes.csv'.format(PRETRAINED_MODEL, LR, BATCH_SIZE, STEPS, EPOCHS))


# # Loading the trained model

# In[ ]:


os.system('ls snapshots')

# In[ ]:


model_path = os.path.join('snapshots', sorted(os.listdir('snapshots'), reverse=True)[0])
model = models.load_model(model_path, backbone_name='resnet50')
model = models.convert_model(model)

# # Predictions

# In[ ]:


li = os.listdir(test_img)
print(li[:5])


# In[ ]:


def predict(image):
    image = preprocess_image(image.copy())
    image, scale = resize_image(image, min_side=512, max_side=512)
    boxes, scores, labels = model.predict_on_batch(
        np.expand_dims(image, axis=0))
    boxes /= scale
    return boxes, scores, labels


# In[ ]:


THRES_SCORE = 0.5


def draw_detections(image, boxes, scores, labels):
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < THRES_SCORE:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(image, b, color=color)

        caption = "{:.3f}".format(score)
        draw_caption(image, b, caption)


# In[ ]:


def show_detected_objects(image_name):
    img_path = test_img + '/' + image_name

    image = read_image_bgr(img_path)

    boxes, scores, labels = predict(image)
    print(boxes[0, 0].shape)
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    draw_detections(draw, boxes, scores, labels)
    plt.figure(figsize=(15, 10))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()


# for img in li:
#    show_detected_objects(img)

# In[ ]:


preds = []
imgid = []
for img in tqdm(li, total=len(li)):
    img_path = test_img + '/' + img
    image = read_image_bgr(img_path)
    boxes, scores, labels = predict(image)
    boxes = boxes[0]
    scores = scores[0]
    for idx in range(boxes.shape[0]):
        if scores[idx] > THRES_SCORE:
            box, score = boxes[idx], scores[idx]
            imgid.append(img.split(".")[0])
            preds.append(
                "{} {} {} {} {}".format(score, int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])))

sub = {"image_id": imgid, "PredictionString": preds}
sub = pd.DataFrame(sub)
sub.head()

sub_ = sub.groupby(["image_id"])['PredictionString'].apply(lambda x: ' '.join(x)).reset_index()

# samsub=pd.read_csv("/kaggle/input/global-wheat-detection/sample_submission.csv")
samsub = pd.read_csv("sample_submission.csv")

samsub.head()

for idx, imgid in enumerate(samsub['image_id']):
    samsub.iloc[idx, 1] = sub_[sub_['image_id'] == imgid].values[0, 1]

samsub.head()

samsub.to_csv('submission.csv', index=False)
