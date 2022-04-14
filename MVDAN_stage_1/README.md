# MVDAN_stage_1
A Pytorch implementation of Multi-view Dual Attention Network for 3D Object Recognitionn(MVDAN)

In all our experiments, there are two stages of training. The first stage only classifies a single view for fine-tuning the model, in which the dual attention block is removed. The second stage of training adds dual attention blocks to train
all the views of each 3D model and performs joint classification for the views, and this is used to train the entire classification framework. When testing, only the second stage is used to make predictions.
