# PyTorch code for MVDAN  
A Pytorch implementation of Multi-view Dual Attention Network for 3D Object Recognitionn[MVDAN](https://link.springer.com/article/10.1007/s00521-021-06588-1) 

In this paper, the 3D object recognition problem is converted to multi-view 2D image classification problem. For each 3D object, there are multiple images taken from different views

### Dependecies

- Python 3.6
- PyTorch 1.2.0
- numpy

### Dataset

- ModelNet CAD data can be found at [Princeton](http://modelnet.cs.princeton.edu/)
- ModelNet40 12-view png images can be downloaded at [modelnet40_images_new_12x (1.6GB)](http://supermoe.cs.umass.edu/shape_recog/shaded_images.tar.gz)  
- You can also create 3-view png images and 6-view png images by reducing the number of 12 views

### Train the model

```python train.py -name MVDAN -num_models 1000 -weight_decay 0.0001 -num_views 12 -cnn_name resnet50```

