# Joint learning of object and action detectors

By Vicky Kalogeiton, Philippe Weinzaepfel, Vittorio Ferrari, Cordelia Schmid 

## Introduction

In our [ICCV17](https://hal.inria.fr/hal-01575804/document) paper we jointly detect objects performing various avtions in videos. 
Here, we provide the evaluation code used in our experiments. 

## Citing Joint learning of object and action detectors

If you find our evaluation code useful in your research, please cite: 

    @inproceedings{kalogeiton17biccv,
      TITLE = {Joint learning of object and action detectors},
      AUTHOR = {Kalogeiton, Vicky and Weinzaepfel, Philippe and Ferrari, Vittorio and Schmid, Cordelia},
      YEAR = {2017},
      BOOKTITLE = {ICCV},
    }

## Evalutation of all architectures

You can download our detection results (multitask, hierarchical and cartesian):
    
    curl http://pascal.inrialpes.fr/data2/joint-objects-actions/JointLearningDetections.tar.gz | tar xz 
    
To run the mAP evaluation function for the multitask, hierarchical and cartesian cases (Table 4 in our [paper](https://hal.inria.fr/hal-01575804/document)), run: 

    vic_A2D_eval(learning_case) # learning_case: 1, 2 or 3 for multitask, hierarchical and cartesian, respectively

## Zero shot learning

To run the mAP evaluation function for the zero shot learning (Table 5 in our [paper](https://hal.inria.fr/hal-01575804/document)), run: 

    vic_A2D_eval_zeroshot 
    
## Object-action semantic segmentation

You can download our semantic segmentation images:
    
    curl http://pascal.inrialpes.fr/data2/joint-objects-actions/JointLearningSegmentationResults.tar.gz | tar xz 
    
To run the semantic segmentation evaluation function (Table 6 in our [paper](https://hal.inria.fr/hal-01575804/document)), run: 

    vic_eval_segmentation 

