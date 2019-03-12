# Motion-based Object Segmentation based on Dense RGB-D Scene Flow
This repository contains the code for our paper on [Motion-based Object Segmentation based on Dense RGB-D Scene Flow](https://arxiv.org/abs/1804.05195). 

## Abstract
Given two consecutive RGB-D images, we propose a model that estimates a dense 3D motion field, also known as scene flow. We take advantage of the fact that in robot manipulation scenarios, scenes often consist of a set of rigidly moving objects. Our model jointly estimates (i) the segmentation of the scene into an unknown but finite number of objects, (ii) the motion trajectories of these objects and (iii) the object scene flow. We employ an hourglass, deep neural network architecture. In the encoding stage, the RGB and depth images undergo spatial compression and correlation. In the decoding stage, the model outputs three images containing a per-pixel estimate of the corresponding object center as well as object translation and rotation. This forms the basis for inferring the object segmentation and final object scene flow. To evaluate our model, we generated a new and challenging, large-scale, synthetic dataset that is specifically targeted at robotic manipulation: It contains a large number of scenes with a very diverse set of simultaneously moving 3D objects and is recorded with a simulated, static RGB-D camera. In quantitative experiments, we show that we outperform state-of-the-art scene flow and motion-segmentation methods on this data set. In qualitative experiments, we show how our learned model transfers to challenging real-world scenes, visually generating better results than existing methods. 

## How to Cite?
```
@ARTICLE{8411477,
author={L. Shao and P. Shah and V. Dwaracherla and J. Bohg},
journal={IEEE Robotics and Automation Letters},
title={Motion-Based Object Segmentation Based on Dense RGB-D Scene Flow},
year={2018},
volume={3},
number={4},
pages={3797-3804},
doi={10.1109/LRA.2018.2856525},
ISSN={2377-3766},
month={Oct},}
```
## Getting Started

### Prerequisites
tensorflow-gpu==1.2.0

### Installing
```
cd segNet2; make
```
## Dataset
[Training Set](http://download.cs.stanford.edu/juno/sceneflownet/BlensorResult_train.tar.gz) <br/ >
[Validation Set](http://download.cs.stanford.edu/juno/sceneflownet/BlensorResult_val.tar.gz)

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
