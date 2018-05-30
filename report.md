# Pedestrian Detection with Faster-RCNN

## 0 Overview

## 1 History and Design Evolution of R-CNN.

Since AlexNet acheieved its great sucess at 2012 in ILSVRC challenge, the application of the Converlutional Neural Network for image classification has dominated the field of both research and industry.  Within this topic, a brief review of the developments on object detection techniques will be presented; region based object detectors including R-CNN, Fast R-CNN, Faster R-CNN and R-FCN will be discussed. To sort out the tech-developments through history will not stay as a review only but also provide a better insight of future.

### 1.0 Before R-CNN: Slideing-windows and Selective-search.

Before all advanced developments in object detection, the solution for applying CNN to object detection  is to slide a windows all over the image and identify objects with classification; and patches from the original images are cut out then to feed in the classification CNN to extract features, inner products to classify and linear regressor for boudiary box.
![sliding-window](images/sliding-windows.png)

ps. To detect various object  at different viewing distances, windows of various sizes and aspect ratios has been used. The patches are then warped to fit the fixed size classfigiers. But this will not impact the classification accuracy since the classifier are trained to handle warped images.

Instead of such a brute method of exhaustion,  a region proposal method to create regions of interest (ROIs) for object detection is rasied as selective search (SS). Individual pixel are grouped by calculating the texture and combine the closest ones. 
![selective-search](images/selective-search.png)
ps. To deal with ovelapping groups, smaller ones are groupped first and then merging regions till everything is combined. 

### 1.1 R-CNN.

[publication](https://arxiv.org/pdf/1311.2524v5.pdf) 
[source code](https://github.com/rbgirshick/rcnn)
At the time of its release, R-CNN is the state-of-the-art visual object detection system; it combines bottom-up region proposals with rich features computed by a convolutional neural network. R-CNN improved the previous best detection performance on PASCAL VOC 2012 by 30% relative, going from 40.9% to 53.3% mean average precision. 

R-CNN applys region proposal method to create about 2000 ROIs (regions of interest) from each training images. And then these ROIs are wraped and feed into a CNN network for classification and booudary box regression.
![rcnn diagram](images/rcnn.png)

Comparing to the sliding-windows solution, the R-CNN takes fewer but higher quality ROIs and thus runs faster and more accurate.

### 1.2 Fast R-CNN.
[publication](https://arxiv.org/pdf/1504.08083.pdf)
[source code](https://github.com/rbgirshick/fast-rcnn)


### 1.3 Faster R-CNN.
[publication](https://arxiv.org/pdf/1506.01497.pdf)
[](https://github.com/rbgirshick/py-faster-rcnn)

The Faster R-CNN 

### 1.4 R-FCN and Other State of Art Developments.


## 2  Faster R-CNN Reimplementation and Analysis of Framework and Key Components.
(*The answer to question 01 is included here: "Please describe the 2 key components in the Faster R-CNN framework"*)

## 3 Reimplement Faster RCNN with Pedestrian Detection Dataset.

The offical faster R-CNN python implementation from [@rbgirshick](https://github.com/rbgirshick) works with `Pascal Voc `dataset while this project aims at pedestrian detecting. Thus, to adapt the existed implementation to pedestrian dataset like `Caltech, Kitti and INRIA` is the essential part for implementing. And in this report, how to prepare the caltech dataset will first be introduced. And how to modify the network model will then be explained.  The last topic will be the performance analysis and demonstration of the trained network.

And this part of work is heavily based on [py-faster-rcnn-caltech-pedestrian](https://github.com/govindnh4cl/py-faster-rcnn-caltech-pedestrian).

### 3.1 Prepare the Dataset: Caltech.

The [Caltech](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/) pedestrain dataset consists of approximately 10 hours of 640x480 30Hz video taken from a vehicle driving through regular traffic in an urban environment. About 250,000 frames (in 137 approximately minute long segments) with a total of 350,000 bounding boxes and 2300 unique pedestrians were annotated. The annotation includes temporal correspondence between bounding boxes and detailed occlusion labels. 

In order to use Caltech dataset for faster R-CNN,  3 essential parts of data need to be obtained/converted from the raw data: Annotations, ImageSets, JPEGImages, which are the label data, text file with images name as lists and images in jpg format respectively. In this project, 8 substeps are implemented to obtain the required data. 


#### 3.1.1 Create directory structure.

Before making the dataset, design and create the directories structure to make it easy for coming data parses and extracts operations.

``` Shell
mkdir caltech;
cd caltech
mkdir downloaded         # Store downloaded setxx.tar
mkdir unzip              # Store unzipped setxx folder
mkdir data               
mkdir data/JPEGImages    # Store prepared images
mkdir data/ImageSets     # Store image name txt file
```

#### 3.1.2 Get the video sequences and annotations.

The dataset can be downloaded from the offical site `https://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/setXX.tar` with commands:

```Shell
wget https://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/set00.tar
wget https://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/set01.tar
wget https://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/set02.tar
wget https://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/set03.tar
wget https://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/set04.tar
wget https://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/set05.tar
wget https://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/set06.tar
wget https://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/set07.tar
wget https://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/set08.tar
wget https://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/set09.tar
wget https://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/set10.tar
wget https://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/annotations.zip
```
Or for this assignment, all tars packages are preloaded to `/DataA/PublicDataSet/LAB2` and can be copied to `caltech/data` with:

```Shell
cp -r /DataA/PublicDataSet/LAB2 .
```

#### 3.1.3 Extract tars into the directory `unzip`.

Extract(unzip) `setxx.tar` to directories with commands:

```Shell
tar -xf downloaded/set00.tar --directory unzip/
tar -xf downloaded/set01.tar --directory unzip/
tar -xf downloaded/set02.tar --directory unzip/
tar -xf downloaded/set03.tar --directory unzip/
tar -xf downloaded/set04.tar --directory unzip/
tar -xf downloaded/set05.tar --directory unzip/
tar -xf downloaded/set06.tar --directory unzip/
tar -xf downloaded/set07.tar --directory unzip/
tar -xf downloaded/set08.tar --directory unzip/
tar -xf downloaded/set09.tar --directory unzip/
tar -xf downloaded/set10.tar --directory unzip/
```

The  `caltech/downloaded` diretory can be removed with command:

```Shell
rm -r downloaded
```
And the directories presents as:

```Shell
caltech/unzip
caltech/unzip/set00
...
caltech/unzip/set10

caltech/unzip/set00/V000.seq
...
caltech/unzip/set00/V010.seq
...
```

#### 3.1.4 Extract the `annotations.zip` file.

Unzip and extract `caltech/annotations.zip` to `caltech/data/annotations` dirctory; and ten `setxx` directory will be placed in `caltech/data/annotations`.

```Shell
cd data
unzip ../annotations.zip .  
```	  
With in all `setxx` directories, there are many `vxxx.vbb`files. The directories now should be:

```Shell
caltech/data/annotations

caltech/data/annotations/set00
caltech/data/annotations/set01
...
caltech/data/annotations/set10

caltech/data/annotations/set00/V000.vbb
...
caltech/data/annotations/set00/V008.vbb
...
```

#### 3.1.5 Parse and extract images to `caltech/data/JPEGImages`.

Convert and parse images from sequences with tool provided by [caltech-pedestrian-dataset-converter](https://github.com/govindnh4cl/caltech-pedestrian-dataset-converter); Clone the converter to data directory and then modify the code by adding the source and target directories (for my case, ../unzip and JPEGImages respectively) to `config.py`.
Script:

```Shell
cd data
git clone https://github.com/govindnh4cl/caltech-pedestrian-dataset-converter
# caltech-pedestrian-dataset-converter will be added to data folder
cp caltech-pedestrian-dataset-converter/scripts/config.py .
cp caltech-pedestrian-dataset-converter/scripts/convert_seqs.py .
# vim config.py
# Update source caltech/unzip and target caltech/data/JPEGImages directories
```
And these two lines in `config.py` need to be modified:

```Python
self.src_base_dir = '../unzip'
self.dst_base_dir = 'JPEGImages'
```
Then run convert_seq.py with command:

```Shell
python convert_seqs.py #convert.py for parsing and extracting images
```

Multiple `setxx` directories will be created in `JPEGImages` and contains multiple `Vxxx` directories which in-turn containing multiple `.jpg` images. The `caltech-pedestrian-dataset-converter` directory can be removed. And the directories presents as:

```Shell
data/JPEGImages

data/JPEGImages/set00
...
data/JPEGImages/set10


data/JPEGImages/set00/V000
...
data/JPEGImages/set00/V014

data/JPEGImages/set00/V000/0.jpg
...
data/JPEGImages/set00/V000/578.jpg
...
data/JPEGImages/set00/V000/1663.jpg
...
```
The modified `config.py` code is attached here:

```python
# config.py
import os

class Config():
  def __init__(self):
      #  ------ User configurable parameters START -----
      # A list of all sets to be parsed
      self.sets = ['set00', 'set01', 'set02', 'set03', 'set04',
                   'set05', 'set06', 'set07', 'set08', 'set09', 'set10']

      # sets = ['set00', 'set01', 'set02', 'set03', 'set04', 'set05'] #train
      # sets = ['set06', 'set07', 'set08','set09', 'set10'] #test

      # print_names_only: If set, does not write files on disk. Just prints their names on console.
      # Used for generating a file containing the names of images relative to the
      # base path(without extension, but includes prefix e.g. 'set01/V000/' )
      self.print_names_only = False  # Default: False

      # Make interval 1 if want to print all file names
      # Used for generating 1x, 10x training/test set for caltech dataset
      if self.print_names_only:
          self.interval = 30

      # src_base_dir should have unzipped 'set00', 'set01' .. 'set09' and 'annotations' directories
      self.src_base_dir = '../unzip'
      # dst_base_dir must exist. Should be empty
      self.dst_base_dir = 'JPEGImages'

      #  ------ User configurable parameters END -----

      self.src_ann_dir = 'annotations'
      self.dst_ann_file = 'annotations.json'  # Output annotation file name

      ## Video dump parameters.
      # Directory where labelled video will be written. Will be created.
      self.video_dump_dir = 'video_dump'
      self.bbox_color = (0, 0, 255)  # (B, G, R)

      if self._test_config() is False:
          print('Configuration Failed. Correct values set in config.py')

  def _test_config(self):
      """
      Tests the validity of configuration
      :return: True if success. Else False
      """
      if not os.path.exists(self.src_base_dir):
          print('Source dir: {:s} does not exist.'.format(self.src_base_dir))
          return False

      if not os.path.exists(self.dst_base_dir):
          print('Destination dir: {:s} does not exist.'.format(self.dst_base_dir))
          return False

      return True
```

#### 3.1.6 Create ImageSets.

Image sets are the text files containing the image names for train and test sets. Modify the `config.py` and convert:

* Set `print_names = 1`
* Edit the `sets` list for respective set names to be included.
* Set `interval` based on whether 1x or 10x set is to be generated for Caltech dataset.

Generation of 1x training set:
(`print_names = 1; sets = ['set00', 'set01', 'set02', 'set03', 'set04', 'set05']; interval = 30;`)

```Shell
$python convert_seqs.py > ImageSets/train_1x.txt
```
Generation of 1x test set:
(`print_names = 1; sets = ['set06', 'set07', 'set08','set09', 'set10']; interval = 30;`)

```Shell
$python convert_seqs.py > ImageSets/test_1x.txt
```
Generation of 10x training set(optional):
(`print_names = 1; sets = ['set00', 'set01', 'set02', 'set03', 'set04', 'set05']; interval = 3;`)

```Shell
$python convert_seqs.py > ImageSets/train_10x.txt
```
The `unzip` directory can be deleted now. And the responding directory presents as:

```Shell
caltech/data/ImageSets
caltech/data/ImageSets/test_1x.txt
caltech/data/ImageSets/train_1x.txt
```
Code for generating imageset txt file; the attached one is for generating train_1x.txt and to generate test_1x.txt uncomment

```shell
# self.sets = ['set06', 'set07', 'set08','set09', 'set10'] #test
```
and comment:

```Shell
self.sets = ['set00', 'set01', 'set02', 'set03', 'set04', 'set05'] #train
```
Code for generating train_1x.txt.

```python
# generate train_1x.txt
# shell command: python convert_seqs.py > ImageSets/train_1x.txt
import os

class Config():
    def __init__(self):
        #  ------ User configurable parameters START -----
        # A list of all sets to be parsed
        # self.sets = ['set00', 'set01', 'set02', 'set03', 'set04',
        #              'set05', 'set06', 'set07', 'set08', 'set09', 'set10']

        self.sets = ['set00', 'set01', 'set02', 'set03', 'set04', 'set05'] #train
        # self.sets = ['set06', 'set07', 'set08','set09', 'set10'] #test

        # print_names_only: If set, does not write files on disk. Just prints their names on console.
        # Used for generating a file containing the names of images relative to the
        # base path(without extension, but includes prefix e.g. 'set01/V000/' )
        self.print_names_only = True  # Default: False

        # Make interval 1 if want to print all file names
        # Used for generating 1x, 10x training/test set for caltech dataset
        if self.print_names_only:
            self.interval = 30

        # src_base_dir should have unzipped 'set00', 'set01' .. 'set09' and 'annotations' directories
        self.src_base_dir = '../unzip'
        # dst_base_dir must exist. Should be empty
        self.dst_base_dir = 'JPEGImages'

        #  ------ User configurable parameters END -----

        self.src_ann_dir = 'annotations'
        self.dst_ann_file = 'annotations.json'  # Output annotation file name

        ## Video dump parameters.
        # Directory where labelled video will be written. Will be created.
        self.video_dump_dir = 'video_dump'
        self.bbox_color = (0, 0, 255)  # (B, G, R)

        if self._test_config() is False:
            print('Configuration Failed. Correct values set in config.py')

    def _test_config(self):
        """
        Tests the validity of configuration
        :return: True if success. Else False
        """
        if not os.path.exists(self.src_base_dir):
            print('Source dir: {:s} does not exist.'.format(self.src_base_dir))
            return False

        if not os.path.exists(self.dst_base_dir):
            print('Destination dir: {:s} does not exist.'.format(self.dst_base_dir))
            return False

        return True
```

#### 3.1.7 The prepared dataset.

After all previous operations, the dataset has been downloaded and parse to  an applicable format and the directory structure presents as:

```Shell
$caltech/data/
$caltech/data/annotations/
$caltechdata/annotations/set00/
$caltech/data/annotations/set01/
$...
$caltech/data/annotations/set10/
$caltech/data/JPEGImages
$caltech/data/JPEGImages/set00
$caltech/data/JPEGImages/set01
$...
$caltech/data/JPEGImages/set10
$caltech/data/ImageSets/train_1x.txt
$caltech/data/ImageSets/test_1x.txt
```

#### 3.1.8 Create symlinks for the dataset.

Then build softlink to the prepared dataset in the `caltech/data` directory with commands:

```Shell
cd caltech/data
ln -s ../caltech caltech
```

### 3.2 Model Tuning to Fit Caltech Dataset.

Like mentioned before this project is based on the `py-faster-rcnn-caltech-pedestrian` project but with many modifications, both code and structure wise.

#### 3.2.0 Clone Repo, Make Caffe and Download pre-trained ImageNet models with Standard Process.

First clone the project from `https://github.com/govindnh4cl/py-faster-rcnn-caltech-pedestrian` and then build caffe.

```shell
git clone --recursive https://github.com/govindnh4cl/py-faster-rcnn-caltech-pedestrian.git
cd py-faster-rcnn-caltech-pedestrian
git checkout master

cd py-faster-rcnn-caltech-pedestrian/lib
make

cd py-faster-rcnn-caltech-pedestrian/caffe-fast-rcnn
cp Makefile.config.example Makefile.config
make -j128 && make pycaffe

cd ../
./data/scripts/fetch_faster_rcnn_models.sh
```

#### 3.2.1 Build Caffe with Support to Python Layer.

If the Caffe is built with the standard process, one issue ([can be refered to offical faster rcnn git issue 31](https://github.com/rbgirshick/fast-rcnn/issues/31))  shows like `Check failed: registry.count(type) == 1 (0 vs. 1) Unknown layer type: Python` will occur and the solution is to modify the Makefile.config file by uncomment `WITH_PYTHON_LAYER := 1` in the Makefile.config and remake caffe.

```shell
make clear
make -j128 && make pycaffe
```
The Caffe is then remade and supports python layers.

#### 3.2.2 Modify the Network to Fit Caltech Dataset.
With the unmodified code, the other issue will be when running the code to 10000 iters, the following error message will pop up and terminate the training: 

```shell
File "./tools/train_net.py", line 112, in <module>
max_iters=args.max_iters)
File "/home/deeplearning_3/Aven_caltech/py-faster-rcnn-caltech-pedestrian/tools/../lib/fast_rcnn/train.py", line 160, in train_net model_paths = sw.train_model(max_iters)
File "/home/deeplearning_3/Aven_caltech/py-faster-rcnn-caltech-pedestrian/tools/../lib/fast_rcnn/train.py", line 111, in train_model model_paths.append(self.snapshot())
File "/home/deeplearning_3/Aven_caltech/py-faster-rcnn-caltech-pedestrian/tools/../lib/fast_rcnn/train.py", line 73, in snapshot self.bbox_stds[:, np.newaxis])
ValueError: operands could not be broadcast together with shapes (84,4096) (8,1)
``` 
This is also an issue can be found on git issues([issue 37](https://github.com/rbgirshick/fast-rcnn/issues/37)) and the solution is relatively complecated. In short, the main issue is caltech dataset has only two classes comparing with 21 from the original pascal voc dataset, the output layers like `cls_score` and `bbox_pred` have to be modifited to fit the dataset; and for this case, the output params for `cls_score` should be 2 and 8 for `bbox_pred` layer. Another further work is to modify the layer name in prototxt and the croresponding class/function name in python layer definition code(I don't fully get the reason but it turns out to be essential).

Three essential files here need to be modified;

* `models/VGG16/faster_rcnn_end2end/train.prototxt`; Change the output params for `cls_score` and `bbox_pred`  layers.
* `models/VGG16/faster_rcnn_end2end/train.prototxt`; Change the name for `cls_score` and `bbox_pred`  layers to a new name like `new_cls_score` and `new_bbox_pred` .
* `lib/train.py`; Change the function name/ parameter name from `cls_score` and `bbox_pred`  to  `new_cls_score` and `new_bbox_pred`.

Everything should be set and the training can be started from here.

### 3.3 Performance and Evaluation: mAp and demos.
The log file is attached at the end of this reprot, and some of the essential info is listed here.
#### 3.3.1 mAP(Mean Average Precision).
(*The answer to question 02 is provided here: "Please describe the object detection performance metric, mAP (Mean Average Precision), and explain why it can well reflect the object detection accuracy."*)

#### 3.3.2 Final mAP performance and demo samples.
(*The answer to question 03 is provided here: "Please train and test the Fast R-CNN framework on one of the existing pedestrian detection datasets, and report the final mAP performance that you have achieved.  The dataset could be Caltech, INRIA, KITTI . Please also report some pedestrian detection examples by including the images and bounding boxes."*)
To demonstrate the train model over sample images on the headless server, the output images can not be displayed but saved . And some  modifications have to be done to the `demo.py` code.

Python code attached here:

```python

```

### 3.4 Implement Faster R-CNN on Kitti Datasets.

## 4. Advanced Approach and Improvement Based on Faster R-CNN
(*The answer to question 04 is provided here: "Propose your own method to further improve the pedestrian detection performance based on the Fast R-CNN framework."*)
