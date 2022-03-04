goal of this project :

Learn the lightning framework. 


# ReCo - Regional Contrast

adaptation of the Reco algorithm with lightning from the following paper :, [Bootstrapping Semantic Segmentation with Regional Contrast](https://arxiv.org/abs/2104.04465), introduced by [Shikun Liu](https://shikun.io/), [Shuaifeng Zhi](https://shuaifengzhi.com/), [Edward Johns](https://www.robot-learning.uk/), and [Andrew Davison](https://www.doc.ic.ac.uk/~ajd/).
(more specificly the semi-supervised with full label procedure)

# how to use :

train_light.py +flags <br />

flags:  <br />
- trainer flags (--gpus=1 for using 1 gpu)
- model flags : "num_segments","lr","weight_decay","apply_reco","apply_aug","weak_threshold","strong_threshold","num_negatives","num_queries","temp","output_dim","optimizer","max_epochs"
- other flags : --path_to_dataset --bakbone

# change regarding the model : 

- model output spatial size is the same for the segmentation head, for the representation head, it is divided by 4
- data augmentation is always done on tensor. I lost the deterministic part in the process
- visualisation : when training the model, we can see the various loss & the images prediction with tensorboard
- EMA model : instead of defining a EMA object, the EMA step is done directly on the pytorch model
- dataset : it is supposed to be divided such as  path/images, path/labels. we pass as arguments the name of the labeled image for the training/validation step
- lightning : using the lightning model means that the default precision is float32 instead of float64

# what I didn't Do :

- In my test, I used 2 classes, and I didn't test for more classes. I did not reimplement the visualisation of the similarity of the different classes (with a tree) such as in the original repository.
- I didn't build a way to use a modular data_augmentation. We can only choose between No aug, applying cutmix, applying classmix, applying cutout.
- didn't fully build a way to change the segmentation model. 

