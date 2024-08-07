# commands
docker run -it --gpus all --shm-size=8g -v $PWD:/code/ -v /home/gpss1/remote/datasets/Wildtrack_dataset:/data/Wildtrack -w /code mvdet

docker run -it --gpus all --shm-size=8g -v $PWD:/code/ -v /home/gpss1/remote/mnts/mnt0:/mnt -v /home/gpss1/remote/datasets/Wildtrack_dataset:/data/Wildtrack -w /code mvdet

python main.py -d wildtrack

python main.py -d wildtrack --cam_adapt --train_viz --resume 2024-06-26_11-16-08

python test.py --log_dir /mnt/default/2024-07-02_09-33-24 --data_path /data/Wildtrack --cam_adapt --trg_cams "2,4,5,6" --cls_thres 0.05 --persp_map

rsync -r erikbro@alvis1:/mimer/NOBACKUP/groups/naiss2023-23-214/mvdet/results/logs/wildtrack_frame/default mnt0/


# Paper 1 outline

## abstract
We consider the problem of unsupervised domain adaptation for mutli-view pedestrian detection. First, we introduce several training tricks that makes the chosen model (MVDet) more generalizable. Second, we show how performance can be furhter increased by adopting self-training techniques that has been widely used for monocular object detection and semantic segmetnation.
- [ ] contributions
  - [ ] propose training techniques that makes multi-view OD networks more generalizable (permutation augmentation, pretraining, dropout, mvaug)
  - [ ] we propose a multi-view self-training framework for UDA of multi-view pedestrian detection. This includes pseudo-labelling in BEV and projecting the pseudo-labels back into each camera view to be able to leverage perspective view and bev view supervision on target data.
  - [ ] extensive evaluation make our paper a first baseline for multi-view pedestrian UDA. It will serve as a baseline that subsequent papers can use to benchmark their proposed improvements to mutli-view UDA. 

## motivation

I'm mainly interested in camera rig adaptation, as this would be very valuable at Volvo. Camera rig adaptation would allow us to collect training data with a single camera rig and then adapt the model to new installations without need for further annotation work.

Therefore, I am interested in the camera rig adaptation benchmarks proposed in the GMVD paper, which include wildtrack->wildtrack, multiviewx->multiviewX and GMVD->GMVD camera rig adaptation. 

Since I aim to use UDA techniques to perform the camera rig adaptation, and the same techniques are also likely applicable to sim2real adaptation, it makes sense to also include such benchmarks to make the paper stronger.
Previously studied sim2real benchmarks include multiviewX->wildtrack and GMVD->wildtrack.

While I like the generalizability and adaptation benchmarks proposed by GMVD, there are several flaws/shortcuts in their implementation/experiments with MVDet.
By using pretraining, random permutations of camera ordering and mvaug, I've been able to significantly improve MVDet on several benchmarks proposed in GMVD.
This is in itself an achievement.
Furthermore, I've successfully implemented UDA self-training for MVDet on some benchmarks.

**How many models should I evaluate?**  
It seems like my "generalizable" MVDet is almost as good as GMVD, so it is definitely relevant to use this model.  
However, if I do not evaluate any other models, there would of course be some critique. Are my methods applicable to transformer architectures? Other CNN based architectures? CNN networks trained with cross-entropy instead of MSE loss? However, I think that the paper could be published even only with this model.

**How many benchmarks should I evaluate?**  
Since the Wildtrack dataset has a very limited test set, and even data leakage between train and test set (some people standing in the same spot), it would be undesirable to only evaluate on wildtrack->wildtrack adaptation.  
I think I'd rather evaluate a single model on many benchmarks, than evaluating many models on a single benchmark.  
Therefore, the next step is to download mutliviewx and GMVD dataset and evaluate my MVDet implementation on also these benchmarks.  
A risk with introducing sim2real adaptation is that there could be further complications, that are not visible when doing real2real camera adaptation. SOlving these issues may not necessarily benefit the original task of real2real camera rig adaptation. This could result in me spending a lot of time on sim2real adaptation while in reality I'm interested in real2real adaptation. However, I find this risk quite low. I suspect that the two tasks are mutually benefitial (i.e. improvements to sim2real adaptation probably also lead to improvements in real2real adaptation).

# Paper 2 outline

## abstract
We consider the task of UDA for multi-view pedestrain detection. We build upon our recent work that proposed a UDA baseline for multi-view object detection. In this paper, we make at least one improvement. For example, we propose to refine the pseudo-labels using multi-objects tracking algorithms. This leads to increased pseudo-label quality and subsequently increased performance of the student network.


# TODO

Evaluate my implemetnation of MVDet on all relevant adaptation benchmarks. If I'm successful in this, I'd say I could write a draft and submit it to a conference even without implementating additional methods. To make the paper stronger, I would implement the proposed techniques also for other models (e.g., MVDetr, SHOT and GMVD).

- [ ] improve MVDet generalization capabilities with training tricks and boost further with UDA.  
These experiments cover table 3 and 4 of GMVD paper, with the addition of 2,4,6->1,3,5 and 1,3,5->2,4,6
  - [x] 2,4,6 -> 1,3,5
    - [x] baseline
    - [x] improve baseline with generalization tricks
    - [x] improve further with uda
  - [ ] 1,3,5 -> 2,4,6 (ONGOING)
    - [x] baseline
    - [x] improve baseline with generalization tricks
    - [ ] improve further with uda, need to continue parameter search
  - [x] 2,4,5,6 -> 1,3,5,7
    - [x] baseline
    - [x] improve baseline with generalization tricks
    - [x] improve further with uda
  - [x] 1,3,5,7 -> 2,4,5,6 (ONGOING)
    - [x] baseline
    - [x] improve baseline with generalization tricks
    - [x] improve further with uda
  - [ ] 1,2,3,4,5,6,7 -> 1,3,5,7
  - [ ] 1,2,3,4,5,6,7 -> 2,4,5,6  
  Before I proceed with 7->4 cam adaptation, I must decide whether to keep the test cams in correct positions or not.

- [ ] improve MVDet on multiviewX -> wildtrack by training tricks and boost further with UDA.  
Pretraining and dropout here could be valuable since multiviewx contains 6 cameras and wildtrack 7.  
These experiments cover table 5 of GMVD paper.

- [ ] improve MVDet on camera rig adaptation on multiviewX. This covers table 6 of GMVD paper



# UDA baseline

## General
- [x] Plot perspective view foot/head predictions on target data in all cameras (not just one as is done now)
- [x] Investigate what component leads to poor generalization (perspective view feature extraction or BEV detection head)
  - [x] Transform perspective view predictins to 3D
  - [x] Do NMS on in bev
  - [x] on target domain: compare bev detections with transformed perspective view detections 
  - [x] draw some conclusion  
      $\exists$ cases where pedestrians are missed in all views $\implies$ persp. view classifier (and/or feature extractor) is inadequate  
      $\exists$ cases where pedestrians are detected in at least one view but missed in bev $\implies$ bev decoder is inadequate  
      I.e., it seems like both persp. view classifier and bev decoder have generalization issues. We want to improve both of these.
- [x] Check why pseudo-labels are very different from teacher predictions after training's finished (see logs 5/7)
  - [x] fixed bug in code
- [x] Check why test scores are different during training and testing (see logs 5/7)
  - [x] fixed bug in code
- [x] run scene generalization exps on e.g., 1,3,5 -> 2,4,6. It makes sense to try with zero cameras overlapping. On the other hand, such scenarios are available in GMVD dataset
- [x] run exps with pretrained resnet18 (simply to set pretrained=True in this repo)
- [x] implement early stopping (saving the model with highest moda and printing best results after training's finished)
- [ ] UDA with confidence weighted cross-entropy
  - [ ] implement cross-entropy loss and train domain generalization network
  - [ ] implement confidence weighted cross entropy in UDA setting

### implement EMA teacher

- [x] EMA teacher
  - [x] implement
  - [x] test ema teacher (train only on supervised and simply keep an EMA teacher on the side. Then test the EMA teacher after training's finished.)





### create pseudo labels
- [x] train with soft labels
- [ ] train with confidence weighted MSE loss
  - [ ] create confidence scores from model prediction which is in range -infty to +infty
- [ ] create pseudo-labels in bev and perspective view separately
  - [x] find probabilities and argmax
  - [ ] non-maximum supression
  - [ ] perspective view 
- [x] train with pseudo-labels
- [x] create pseudo-labels in bev and project into perspective view
  - [x] find the pos of pseudo-labels in bev
  - [x] project pos to cameras
  - [x] plot pseudo-labels during training

The model predicts both head/feet positions in each image as well as occupancy map in bev. 
It seems natural that the student should be supervised in both perspective and bev view also on target data.

1. use the teacher's soft labels (predictions) in both perspective and bev view.
    For this option, I should not run a gaussian kernel over the teacher predictions, as this will merely make them even more uncertain.
2. convert teacher's predictions to pseudo-labels in both perspective and bev view
    Here, it could make sense to run the gaussian kernel (treat the pseudo-labels exactly like real labels)
3. create pseudo-labels only in bev. Then project these labels into the images and supervise with pseudo-labels both in perspective and bev.
    Again, it makes sense to run the gaussian kernel.

Assuming that training with pseudo-labels is more beneficial and that the preds in bev view are more accurate than those in image view, option 3 should be most favourable.

Option 1 has a natural "confidence weighting" as we use the MSE loss. I.e., for a teacher detection with ~0.7 confidence, the loss will not be as large if the student is incorrect as it would have been for a teacher detection with ~1 confidence.
For option 2 and 3, it could make sense to introduce confidence weighting to reduce the impact of noisy regions. This should be easy to do with a simple weighted MSE loss, where the weight is chosen as the confidence.

### data augmentation
- [x] dropview (GMVD propose to always drop one camera. I believe that it could be better to sometimes include all cameras, e.g., set a proability of dropping one camera)
  - [x] Train baseline cam_adapt with dropview on source (no uda)
  - [x] Train UDA self-training with dropview on source and target
- [x] permutation augmentation (change the ordering of cameras). This could make the BEV decoder less overfit to a specific camera rig. 
- [ ] 3DROM
- [x] MVAug

Strong data augmentation should be used for the student.
Different options exist.

1. use 3D random occlusion (introduced by 3DROM)
2. use MVAug data augmentation (warping of images)
3. use drop-camera (GMVD), i.e., student sees fewer cameras than the teacher. An option here is to supervise the student in perspective view in all cameras, but drop one as it creates the bev predictions. Note that when GMVD introduced the dropview augmentation, they could simply skip a camera with their architecture since they use average pooling. However, for e.g., MVDet, they must process the dropped view since the architecture doesn't allow for decreasing the number of cameras. A natural choice is to set the dropped view to all zeros, however, GMVD chose to duplicate one of the other views instead. I feel like this should result in a "false" training signal as there is a risk of fooling the network. It seems much better to set it to all zeros, which is what I will do.


### ramp-up adaptation
- [x] target loss weight increases as confidence of pseudo-labels increase
  - [x] Since there is no obvious method for measuring the model's confidence (it outputs real values and tries to match the ground truths which has been gaussian smoothed), I resort to a hard-coded schedule for progressively increasing the focus on target domain.

### other UDA concepts
- [ ] ImageNet feature distance as introduced by DAFormer


There should probably be more focus on accurate source labels in the beginning, and then successively focus is shifted to target domain as the quality of the pseudo-labels increase.


# Experiments

### all cameras MVDet baseline
/mimer/NOBACKUP/groups/naiss2023-23-214/mvdet/results/logs/wildtrack_frame/default/2024-07-01_18-07-44
slurm-2464253_3  
moda: 87.4%, modp: 75.5%, precision: 93.2%, recall: 94.2%

### cam_adapt 1,3,5,7 -> 2,4,5,6
slurm-2465377_5  
/mimer/NOBACKUP/groups/naiss2023-23-214/mvdet/results/logs/wildtrack_frame/default/2024-07-02_10-44-47




### cam_adapt 2,4,5,6 -> 1,3,5,7
slurm-2465263_4
/mimer/NOBACKUP/groups/naiss2023-23-214/mvdet/results/logs/wildtrack_frame/default/2024-07-02_09-33-24

testing on 2,4,5,6 (test_1)  
moda: 83.5%, modp: 72.8%, precision: 94.7%, recall: 88.4%  
(Results from GMVD paper: 85.2, 72.2, 92.6, 92.)


testing on 1,3,5,7 (test_0)  
moda: 18.2%, modp: 70.2%, precision: 76.6%, recall: 26.2%  
(Results from GMVD paper: 43.2, 68.2, 94.6, 45.8)


Making predictions on training dataset to see the "pseudo-labelling capability" of the model
cls_thresh=0.05 2024-07-02_09-33-24/test_13  
cls_thres=0.2 test_12  
cls_thres=0.4 test_11  
As expected, the quality of pseudo-labels is relatively poor. cls_thres=0.2 seems most reasonable out of the three.


It can be seen that my experimental results match those of the GMVD paper relatively well, although, the moda and recall is a bit lower than expected on 1,3,5,7.

### verifying EMA teacher on 2,4,5,6->1,3,5,7
Only training on source, but updating the ema throughout training.  
experiment folder: /mnt/default/2024-07-05_09-48-35  
using cls_thres=0.4 in all the below experiments

Performance of "student" on 2,4,5,6 (test 0)
moda: 82.9%, modp: 73.4%, precision: 91.3%, recall: 91.6%

Performance of ema on 2,4,5,6   (test 1)
moda: 81.1%, modp: 73.1%, precision: 89.1%, recall: 92.3%

The ema performance similarly to the student, but not exactly the same, which is expected. => EMA implemetnation seems OK.


is it beneficial to use ema in this setting?
From below experiments, it doesn't seem like ema by default has better generalization capabilities.

Performance of "student" on 1,3,5,7 (test4)  
moda: 18.1%, modp: 70.5%, precision: 70.8%, recall: 30.8%  

Performance of ema on 1,3,5,7 (test 3)  
moda: 18.7%, modp: 68.9%, precision: 68.9%, recall: 34.1%


### student-teacher soft labels (no augmentation) on 2,4,5,6->1,3,5,7
/mnt0/default/2024-07-05_17-02-52-550235  
slurm-2478540_15  
target_weights:  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8135804472781414, 0.9238205812014586]


2,4,5,6 scores (available in test_0):  
moda: 80.8%, modp: 73.2%, precision: 88.1%, recall: 93.4%

1,3,5,7 scores  (available in test_1):  
moda: 21.6%, modp: 69.3%, precision: 72.0%, recall: 35.4%

Seems like student-teacher training with soft-labels may give very slight performance boost, or at least doesn't hurt performance.  
Maybe the performance boost will come when I introduce augmentation.   

### student-teacher pseudo-labels (no augmentation) on 2,4,5,6->1,3,5,7
/mnt/default/2024-07-05_16-57-33-979160  
slurm-2478540_2  
target_weights:  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.18508207817320688, 0.9442114246448788]  
pseudo_label_th:  0.38431918345836336  


1,3,5,7 scores  
cls_thres=0.4 (available in test_0)  
moda: 12.6%, modp: 70.1%, precision: 88.2%, recall: 14.5%  

cls_thres=0.3 (available in test_1)  
moda: 18.2%, modp: 69.8%, precision: 80.4%, recall: 24.1%  

cls_thres=0.2 (available in test_2)  
moda: 22.0%, modp: 68.9%, precision: 69.6%, recall: 39.0%


Also seems like an okay result.  But still no clear improvements.

Note that all the 9 other similar experiments lead to very poor performance with ~90 % recall and ~0 % precision. The reason seems to be that the pseudo-label threshold was set too low, resulting in many false positive pseudo-labels. The conclusion is that the pseudo-label threshold should probably be at least ~0.35. Perhaps 0.35-0.45 is a reasonable range (although the maximum value I tried in the previous 10 experiments was 0.38).


### verifying dropview  2,4,5,6 -> 1,3,5,7
previous experiment 2024-07-02_09-33-24 (without dropview) gets test-scores on 1,3,5,7 without and with dropview during test-time:

moda: 83.5%, modp: 72.8%, precision: 94.7%, recall: 88.4%  
moda: 46.9%, modp: 71.3%, precision: 94.4%, recall: 49.9%

New experiment 2024-07-09_09-32-28-836149 gets the scores

moda: 74.4%, modp: 70.7%, precision: 86.9%, recall: 87.6%  
moda: 56.2%, modp: 69.7%, precision: 88.2%, recall: 64.9%

It can be seen that the regular test-score is a bit worse after training with dropview. This is reasonable since using dropview during training may make it harder for the model to exploit information from all 4 views simultaneously.  
However, after training with dropview, we see that the performance doesn't drop as much when using test-time dropview.  
Conclusion: dropview seems to work. However, it is probably not beneficial to use dropview in every iteration, as this may make it more difficult for the model to exploit all views during test time.


### verifying permutation augmentation 2,4,5,6 -> 1,3,5,7
Two exps training only on soruce data:
one where the permutation is merely change (permutation = [3,0,2,1]), should yield similar performance as long as cam 5 is in the same position.  
Another where the permutation augmentation is used (new permutation every iteration). Now, performance on 1,3,5,7 will probably drop significantly as camera 5 is not always in the same place.

With permutation=[3,0,2,1]:  
moda: 32.6%, modp: 67.5%, precision: 89.1%, recall: 37.1%  

with random permutation:  
moda: 44.0%, modp: 67.8%, precision: 92.8%, recall: 47.7%

Conclusion: permuattion seems to be implemented correctly. And it seems beneficial with random permutation even though camera 5 is on the same place in train and test.


### verifying MVaug augmentation 2,4,5,6 -> 1,3,5,7
Training with mvaug augmentation (only 30 degrees though, and no scale/sheer)

On 1,3,5,7:  
moda: 30.5%, modp: 66.2%, precision: 90.5%, recall: 34.0% (with incorrect parameters of raff, i.e. only 30 % rot)

moda: 40.0%, modp: 65.3%, precision: 94.8%, recall: 42.3% (with GMVD parameters of raff, i.e. rot 45% etc, although 100% augmentation probability)


On 2,4,5,6:  
moda: 76.8%, modp: 65.2%, precision: 92.5%, recall: 83.5%

Conclusion: Performance on 1,3,5,7 is slightly better than the baseline, while performance on 2,4,5,6 is slightly worse. So, while the augmentation makes the model generalize better, it makes it a bit more difficult to fit to the training data. Seems reasonable.

Notes: since precision was really high while recall was low, I also tried lowering the cls_thres to 0.2, which resulted in predictions seen in the image. The main issue is that there are a lot of false positives right at the edge of the bev map, which probably comes from the fact that there are a bunch of people sitting and standing there, as seen in the perspective image of camera 7. It could be that if the bev view was larger, then the nms would remove these predictions as the maximum actually lies outside the region of interest. It seems reasonable that any model based on NMS a prone to having issues at the boundaries. However, I suppose one could also argue that these points shouldn't receive so high scores in the first place.  
![](resources/images/map_13.jpg)  
![](resources/images/output_cam7_foot_33.jpg)

### MVAug + random permutation 2,4,5,6 -> 1,3,5,7

On 1,3,5,7:  
moda: 53.7%, modp: 63.1%, precision: 92.2%, recall: 58.6%

Conclusion: both random permutation and mvaug improve generalization capabilities individually, but the combination of the two yields the highest performance.



### generalizable exp (pretrained, mvaug, permutation) 2,4,5,6 -> 1,3,5,7
2024-07-16_10-20-24-986906

On 2,4,5,6  
moda: 81.6%, modp: 70.8%, precision: 95.9%, recall: 85.3%

On 1,3,5,7  
moda: 67.3%, modp: 68.8%, precision: 96.1%, recall: 70.2%

One of the failure cases on the test ( 1,3,5,7) set is shown below. While the five-people-group is quite clearly detected in cam7, the score in bev for this group of people is low, so none of them are detected. It so happens that these people are partly occluded in cam5 in this frame, which may be the reason they are not detected.  


Concluding remarks: in this case, it may be difficult to construct reliable pseudo-labels in bev because the detections are very uncertain. On the other hand, the detections in cam7 seems reliable, so it may be a good idea to use these detections in self-training. Furthmore, we can conclude that since the detections in cam7 are of high quality, surely the image features of cam7 is also of high/decent quality. As such, the bev features derived from these image features are also of good quality (since we merely perform bilinear sampling). The poor bev predictions is thus a result from inadequate decoding of the bev features. It could be that the bev decoder is fooled by poor features from the other cameras, or that it simply doesn't deem the high quality features from cam7 to provide enough evidence for the detections.

Hypotheses:  
1. The bev predictions are typically of lower quality than the perspective view predictions. 
2. Sometimes the bev predictions are better, and sometimes the perspective view predictions are better.
3. The bev predictions are typically of higher quality than the perspective view predictions.


Possible methods to perform UDA under hypotheses:
1. it is best to use perspective view predictions for self-training. We could project the soft-labels of each camera to bev and create a single bev soft label by either averaging or max pooling. Soft bev label can then be projected to each camera view. Or we could create hard labels at some point
2. It is best to use a combination of perspective and bev view predictions for self-training. We could use a heuristic method to fuse the perspective view predictions with the bev predictions. E.g., project perspective view soft/hard preds to bev and do averaging or max pooling together with the bev predictions. The combination of both perspective view and bev predictions may be more reliable than either one separately.
3. it is best to only use the bev predictions for self-training. Create soft or hard labels using bev predictions. 

An option to smoothly cover all three cases with a single hyper parameter could be to create two separate loss terms, i.e. $(1-\alpha)*L_{persp} + \alpha*L_{bev}$ and let $L_{persp}$ be the loss derived from soft/hard labels in bev produced by perspective view predictions, and $L_{bev}$ be the loss derived from soft/hard labels in bev produced by bev view predictions. Then I could adjust $\alpha$ from 0 to 1 to address hypothesis 1,2 and 3.


![](resources/images/map_24.jpg)
![](resources/images/output_cam7_foot_24.jpg)
![](resources/images/output_cam5_foot_24.jpg)


### wildtrack 2,4,6 -> wildtrack 1,3,5
2024-07-16_15-16-37-676024

| pretrained | permutation | mvaug | dropview | scores                                                                                 |
| ---------- | ----------- | ----- | -------- | -------------------------------------------------------------------------------------- |
| -          | -           | -     | -        | max_moda: 10.8%, max_modp: 41.4%, max_precision: 85.0%, max_recall: 13.1%, epoch: 2.0% |
| x          | x           | x     | x        | max_moda: 54.2%, max_modp: 65.5%, max_precision: 89.6%, max_recall: 61.3%, epoch: 6.0% |

Like the experiment on 2,4,5,6 -> 1,3,5,7, the group of five people is not detected in frame 24 by the bev decoder.  
Additioanlly, none of the perspective view predictions provide confident prediction of this group of people (probably due to the heavy occlusion at this point in time). This was not the case when using 1,3,5,7 as then camera 7 provided quite confident detections. 
So on 2,4,6 -> 1,3,5, there are definitely cases where neither perspective views nor bev decoder can detect certain people.  
We can probably conclude that it is due to inadequate feature extraction in the image plane that the group of people cannot be detected under occlusion. As they are in fact partly visible in two of the cameras, it seems sensible that they could be detected given more training on such difficult partly occluded samples. 
There exist many frames where the bev predictions projected into some of the cameras could give supervision on such partly occluded samples, so here self-training on bev predictions could make sense.




### generalization summary
2,4,5,6 -> 1,3,5,7  
| pretrained | permutation | mvaug | dropview | scores                                                                                  | save_dir                   |
| ---------- | ----------- | ----- | -------- | --------------------------------------------------------------------------------------- | -------------------------- |
| -          | -           | -     | -        | max_moda: 28.9%, max_modp: 66.6%, max_precision: 95.4%, max_recall: 30.4%, epoch: 4.0%  |                            |
| x          | -           | -     | -        | max_moda: 41.1%, max_modp: 70.3%, max_precision: 98.5%, max_recall: 41.7%, epoch: 5.0%  |                            |
| -          | x           | -     | -        | max_moda: 59.0%, max_modp: 66.9%, max_precision: 93.0%, max_recall: 63.9%, epoch: 5.0%  |                            |
| -          | -           | x     | -        | max_moda: 44.5%, max_modp: 67.0%, max_precision: 90.3%, max_recall: 49.9%, epoch: 9.0%  |                            |
| -          | x           | x     | -        | max_moda: 56.4%, max_modp: 66.4%, max_precision: 93.5%, max_recall: 60.6%, epoch: 10.0% |                            |
| x          | x           | x     | -        | max_moda: 67.3%, max_modp: 68.8%, max_precision: 96.1%, max_recall: 70.2%, epoch: 9.0%  | 2024-07-16_10-20-24-986906 |
| x          | x           | x     | x        | max_moda: 65.7%, max_modp: 68.8%, max_precision: 95.4%, max_recall: 69.0%, epoch: 10.0% |

1,3,5,7 -> 2,4,5,6   
| pretrained | permutation | mvaug | dropview | scores                                                                                  |
| ---------- | ----------- | ----- | -------- | --------------------------------------------------------------------------------------- |
| -          | -           | -     | -        |                                                                                         |
| x          | -           | -     | -        |                                                                                         |
| -          | x           | -     | -        |                                                                                         |
| -          | -           | x     | -        |                                                                                         |
| -          | x           | x     | -        |                                                                                         |
| x          | x           | x     | -        | max_moda: 59.9%, max_modp: 64.0%, max_precision: 92.0%, max_recall: 65.5%, epoch: 7.0%  |
| x          | x           | x     | x        | max_moda: 54.0%, max_modp: 64.1%, max_precision: 96.7%, max_recall: 55.9%, epoch: 10.0% |


Notes: my best results are competitive with GMVD, although still slightly worse. But they are **much** better then the results reported on MVDet by GMVD.

Although both permutation and mvaug bring improvements separately (when not using pretrained weights), permuatation+mvaug is alightly worse than only permutation. However, while max_moda is achieved at epoch 5 with only permutation, it is achieved at epoch 10 with permutation+mvaug, suggesting that the model may not have reached it's max performance yet.
=> I should increase the number of epochs slightly to allow for longer trainings when doing aggressive data augmentation.



wildtrack 2,4,6 -> wildtrack 1,3,5
| pretrained | permutation | mvaug | dropview | scores                                                                                 |
| ---------- | ----------- | ----- | -------- | -------------------------------------------------------------------------------------- |
| -          | -           | -     | -        | max_moda: 10.8%, max_modp: 41.4%, max_precision: 85.0%, max_recall: 13.1%, epoch: 2.0% |
| x          | x           | x     | x        | max_moda: 54.2%, max_modp: 65.5%, max_precision: 89.6%, max_recall: 61.3%, epoch: 6.0% |


In my experiments, I've discovered different types of failure cases:
1. pedestrians are missed in all cameras and in bev (e.g. group of five in frame 24 on 2,4,6->1,3,5)
2. pedestrians are missed in bev, but detected in at least one camera (see above section of  2,4,5,6 -> 1,3,5,7)
3. pedestrian is detected in bev, but "more or less" missed in all cameras (i.e., no camera alone provides strong evidence for the detection, .e.g woman in brown cote, one of the most right-most detections in frame 7 of 2,4,6->1,3,5)

Here, we could say that failure 1. probably is due to inadequate feature extraction in perspective view. We need to improve the perspective view feature extraction to handle these errors.  
Failure 2 is due to inadequate fusion of the image-view features. I.e., while image view feature extraction is "good enough", the bev fusion is not well adapted and produces the error. We need to make the bev decoder better to handle these errors.  
Failure 3 indicates a capable bev decoder, while the poor image features makes detection difficult. We could probably benefit from better image view feature extraction. 


Other failure cases include:
1. detections are inaccurate in all cameras and in bev (e.g., two pedestrians are not clearly separated and may appear as one detection, e.g. frame 14 of 2,4,6->1,3,5)
  


# Notes
- Camera C3 is not undistorted properly. Perhaps they use another cameramodel for this camera? The projection of points looks alrgiht, although lines does not appear straight in this camera.
- Isn't it strange to evaluate 2,4,5,6->1,3,5,7 since camera 5 is avialable (and in the same ordering) in both camera rigs? Perhaps we are basically just evaluating the models "single camera" performance, using only camera 5, while cameras 1,3,7 are useless.
- GMVD uses resnet18 pretrained with ImageNet. In this repo, no pretrained weights are loaded originally. However, we can easily set pretrained=true when building resnet18. Then weights will be loaded from 'https://download.pytorch.org/models/resnet18-5c106cde.pth' , but it doesn't say what type of weights this is (imagenet?).
-  How are predictions outside the platform treated? Are they simply ignored or do they lead to lower precision since they are not in the labels?  
- predictions close to the boarder of the bev grid (region of interest) are problematic since nms doesn't really work on the boarder. If we would extend the predictions bev grid such that it is larger than the evaluation bev grid, it could be that nms finds that predictions that where originally inside the bev roi, would instead be outside.

### Tracking
GMVD introduces new benchmarks for evaluating the generalizability of multi-view detectors. They evaluate MVDet, MVDetr, SHOT and GMVD on these benchmarks.
However, they haven't adopted any UDA/SSL techniques.

If I am to use the GMVD benchmarks, it makes sense that I compare with their paper.
One proposal is to:
1. Apply student-teacher self-training to MVDet, MVDetr, SHOT and GMVD, to see if the results from GMVD paper can be improved. Perhaps even the ordering of the results will change? I.e., while some methods are better at directly generalizing, they may not compare as favorably after adaptation.
2. Further investigate how tracking can be incorporated in student-teacher self-training. The idea is that tracking can reduce the noise in the pseudo-labels.
3. Alternatively, investigate design of architecture or training specifics to make the bev-features more generalizable. Perhaps adversarial training can be used to achieve "domain-invariant" BEV features? Or maybe unsupervised training with some MAE variation can be used?

Student teacher self-training will be applied in an UDA setting, where labels are available for some cameras (training) and unavailable for the testing cameras. It also makes sense to evaluate it for sim2real adaptation. Maybe that is enough of a scope? I can skip introducing semi-supervised benchmarks, and I can perhaps also skip introducing more unlabeled data.
However, if I use the same data for training cams and testing cams, the labels could easily be propogated from train to test cams. If I want to avoid this, I should perhaps use other unlabeled data for the test cameras.

Can I apply some standard tracking methodology to MVDet, MVDetr, SHOT and GMVD? The CV-LAB at EPFL implements the min-cost max flow algorithm MuSSP for MOT that e.g., MVFlow used in their paper. Seems like tracking is performed only based on detections in 3D, using the location and probability of detection. Seems easy enough to implement and adopt for other methods.
In MVFlow, they already evaluated MVDet + MuSSP and MVDetr + MuSSP, which performs quite well. 

MVFlow, MVDet, MVDeTr, GMVD are implemented in pytorch.


I wonder how missed detections are handled in the min cost max flow formulation? Can a track skip a timestep and continue later?




# log book
### 1/7
started baseline experiments

### 2/7
The experiemnts yesterday didn't turn out well. The model cannot generalize. From the appearance of the predictions, it seems like the wrong calibration matrices are used.  
Indeed, it seems like the code is incorrect.  Iä've fixed this and started new experiments on cam_adaptation.  

After above fix, the model generalizes almost equally "well/poorly" as described in GMVD, so it seems like the implementation is correct.

I've started experimenting with self-training, but without good results yet.
It is evident that after training only on source, the pseudo-labels on target are of poor quality.
This is expected, and given the results of "Toward unlabeled multi-view 3D pedestrian detection by generalizable AI: techniques and performance analysis", I don't believe that it is worth attempting a naive iterative pseudo-labeling training.
Instead, it is time to implement the mean teacher and experiment with a "smooth" transition to the target data.


### 3/7
The experiement with the ema teacher didn't lead to good performance, but the implementation seems to work.
Before starting loads of experiments on the EMA self-training to find good hyperparameters (i.e., EMA and target weight schedule), I should probably verify tthat the ema teacher works.

I should also figure out whether it is the perspective view backbone or the BEV detection head that has poor generalization capabilities. I would expect that the (ImageNet pretrained) backbone can generalize to new images fairly well, while the BEV detection head overfits to the specific camera setup. An adaptation strategy would in this case involve ensuring that the BEV feature map and detection head becomes more general.

One idea is to use **Domain Invariant Feature Learning** to learn BEV features that are invariant to the camera rig.
- Assumption: Projection to ground plane results in bev features that have appearance heavily dependant on the camera setup (since e.g. pedestrians are smeared out on the floor differently based on the camera angle).
- Applying a few conv layers onto the BEV projection and then enforcing domain invariance on the features could perhaps lead to more "true" BEV features, where the pedestrians are well localized and look like they are actually viewed from above. This is reasonable since a "true" BEV view is independant of the camera rig.
- It is critical that the above (presumably domain invariant) features are also used for subsequent detection. Training on source labels for detection jointly with the above adversarial training can assure that the features do not collapse to nonsense, since the detection supervision will enforce rich features.

Another idea to attain BEV features that are more or less independent of the camera rig is to use the method presented in MVTT. Here, they first use bounding boxes to aggregate perspective view features and then project this feature vector (a single vector per pedestrian) onto a sparse BEV feature map. The drawback with this is that the feature aggregation is very much determined by single view detection performance.

### 4/7 
Created BEV predictions by only using perspective view detections. I.e., make detections in perspective view -> project them all to bev -> NMS to get final bev predictions.

These predictions are competitive with the standard BEV predictions in the cam adaptation setting. However, they are not better, so it is not clear whether the domain gap lies mainly in the perspective view backbone or in the bev decode head.  
There are a few problems with the evaluation above.  
First, since the perspective view is trained to detect feet (rather than pedestrians), it cannot detect any pedestrian that is too close to the camera.  
Second, the above scheme tests also the perspective view to bev projection and sensor fusion algorithm. It is not a precise evaluation of the perspective view prediction quality.  
It would perhaps be better to evaluate the adaptation capabilities of the perspective view backbone **in** the perspective view.
Alt 1. evaluate perspective view detections per camera.  
Alt 2. evaluate the joint perspective view detections (for example, are there any pedestrian that is missed in all cameras?)  

When evaluating different camera setups, it would be helpful to plot the field of views of each camera in the bev map to understand which parts of the bev we can expect detections in.  

After training on "2,4,5,6" the output on cam2 looks like below.  
It can be seen that the network doesn't provide very confident predictions in perspective view even on the cameras that are included in the train set.
![alt text](resources/images/output_cam2_foot_38.jpg)

When evaluating on "1,3,5,7", the predictions in camera 3 looks like below.  
The quality seems to be similar to the predictions in the training set.  
To confirm this quantitatively, maybe I should just print the loss? Rather than printing MODA/MODP, as this would require me to set thresholds and do NMS.
![alt text](resources/images/output_cam3_foot_28.jpg)

### 5/7
When running student teacher self-training with  
dropview :  False  
soft_labels :  False  
target_epoch_start:  6  
target_weight_start:  0.8442657485810173  
target_weight_end:  0.9778772670996941  
pseudo_label_th:  0.3541755216352376  

folder: 2024-07-05_15-21-12-552959  
slurm-2478176_0  


The predictions during training looks very strange (see below)  
![](resources/images/train_target_map_39.jpg)
![](resources/images/foot_pseudo_label_cam1_159.jpg)

But when evaluating the final model, the predictions looks quite alright  
![](resources/images/map_9.jpg)

preds of ema teacher on test set are available in test_3  
preds of ema teacher on train set are available in test_4  
The ema teacher preds on train set in test_4 doesn't match the pseudo-labels produced during training at all... Strange! Seems to be something wrong with the implementation.  

Also, I don't yet understand why test scores (MODA/MODP etc) evaluated during training are very differetn from the same scores produced by test.py.  

### 8/7

Training with pseudo-labels or soft labels yields decent performance in some experiments.  
Pseudo-label threhsold should be ~0.35-0.45 (at least no less than 0.35, as this yhields many false positives).  

Time to run similar experiments with data augmentation.

**Why do we compute precision and recall in two different ways in test?**  
The test script report results in the below format:  
moda: 20.8%, modp: 69.5%, precision: 67.0%, recall: 41.0%  
Test, Loss: 0.007522, Precision: 2.6%, Recall: 39.1,    Time: 33.777  
where the first line gets the result from evaluatedetection.py, which uses NMS and hungarian algorithm for matching, while the second line simply sets as cls_thres  
and computes precision and recall without nms or hungarian algorithm. Typically, many pixels/squares exceed  the cls_thres, resulting in many false positives.

### 9/7
There was a bug in the code, resulting in the source camera matrices where used also when training on target data in the UDA setting.  
This resulted in bad pseudo-labels, as seen in 2024-07-05_15-21-12-552959 .  
After fixing this, the pseudo-labels makes more sense: 2024-07-08_18-12-16-463172 .  
Also, the test scores during training/test are the same now. So a lot of progress today!

How are predictions outside the platform treated? Are they simply ignored or do they lead to lower precision since they are not in the labels?  
Any good UDA technique is probably expected to detect the pedestrians outside the region of interest as humans, so I think that they should be ignored in the evaluation.  
Or perhaps it is okay to supervise the target domain not to predict pedestrians in this region, for fair evaluation?


### 10/7 working on implementing MVAug

in frameDataset, the self.transform includes a resize(720, 1280), which makes the loaded images smaller. However, I don't find anywhere in the code that the projection matrices
are adjusted because of this.
I believe that my current visualization of bev-image is slightly wrong due to this scaling 1920/1280 = 1.5.
Maybe this doesn't matter for the MVDet code since they anyway normalize the image coordinates to [-1, 1] in kornia.warp_perspective, but it may cause issues for me.

I have succeded in warping the input images as well as the foot gt coordinates so that they align with the warped image.

I've also warped the bev label, but I'm not sure if this is done correctly since they seem to use an entirely different technique in MVAug.

I've also figured out that while MVDet uses a projection matrix for image -> bev, MVAug uses a projection matrix for bev -> image.
It is essential that I use the MVAug matrix for the MVaug augmentations, otherwise things won't work.

TODO
- [x] successfully create bev images for unaugmented and MVaugmneted images
- [x] apply the same bev-projection to image features instead of RGB image and check results
- [x] repeat the above two steps now also using scene augmentation


### 15/7
TODO:  
if mvaug is not used, the proj_mats in proj_mats_mvaug_features will not be inverted, i.e., subsequent calls to mthe model will not work.  

MVAug implementation seems correct now.  
It is time to start doing some experiments:

- [x] generalization experiment (only mvaug, compare with only random perm)
- [x] generalization experiment (mvaug + random perm) 

### 16/7
- [x] only 50% of data should be augmented with MVAug according to GMVD
- [x] implement early stopping (saving the model with highest moda and printing best results after training's finished)
- [x] implement weak aug for teacher and strong aug for student (weak mvaug  + random perm for teacher, strong mvaug + random perm + dropview for student)
- [x] run exps with pretrained resnet18 (simply to set pretrained=True in this repo)
- [x] implement dropview probability

Should I examine other adaptation benchmarks before trying to device an adaptation strategy. Yes, I believe that is a good idea. Study a few more relevant benchmarks to select the hypothesis which is most likely to be true. Then I will device an UDA method to address the chosen hypothesis.

- [ ] multiviewx -> wildtrack. The performance of MVDet reported in GMVD on this benchmark is low, but I believe I can pump it up with my augmentation techniques.
- [ ] multiviewx -> multiviewx as proposed by SHOT
- [x] wildtrack 2,4,6 -> wildtrack 1,3,5

### 17/7
- [x] uda trainings on 2,4,6->1,3,5


baseline:   
max_moda: 68.9%, max_modp: 66.7%, max_precision: 91.8%, max_recall: 75.6%, epoch: 8.0%


uda experiments:  
target_weights:  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.529442357047416, 0.5537716176786276, 0.5781008783098391, 0.6024301389410507]
pseudo_label_th:  0.4196905898765535
max_moda: 65.5%, max_modp: 62.2%, max_precision: 98.0%, max_recall: 66.9%, epoch: 8.0%

target_weights:  [0.0, 0.0, 0.0, 0.47958319143093275, 0.5507580172662846, 0.6219328431016363, 0.6931076689369882, 0.7642824947723399, 0.8354573206076917, 0.9066321464430436]
pseudo_label_th:  0.39977931854928195
max_moda: 68.8%, max_modp: 64.3%, max_precision: 95.5%, max_recall: 72.2%, epoch: 9.0%

target_weights:  [0.0, 0.0, 0.0, 0.08801375805758505, 0.14867617584590007, 0.2093385936342151, 0.27000101142253013, 0.33066342921084513, 0.3913258469991602, 0.4519882647874752]
pseudo_label_th:  0.41709663880397635
moda: 66.0%, modp: 62.2%, precision: 96.4%, recall: 68.5%
max_moda: 69.6%, max_modp: 62.7%, max_precision: 94.3%, max_recall: 74.2%, epoch: 6.0%

target_weights:  [0.0, 0.0, 0.5656575015469459, 0.6174317863351922, 0.6692060711234386, 0.720980355911685, 0.7727546406999313, 0.8245289254881776, 0.8763032102764239, 0.92807749506]
pseudo_label_th:  0.43602802542276353
max_moda: 60.0%, max_modp: 61.9%, max_precision: 98.6%, max_recall: 60.8%, epoch: 8.0%

target_weights:  [0.0, 0.370242316115, 0.38931753997, 0.40839276384, 0.4274679877, 0.4465432115647331, 0.46561843542701653, 0.4846936592893, 0.5037688831515835, 0.522844107013]
pseudo_label_th:  0.3942720007566877
max_moda: 32.1%, max_modp: 58.6%, max_precision: 91.8%, max_recall: 35.3%, epoch: 2.0%

target_weights:  [0.508953068, 0.52522553, 0.541498005, 0.55777047, 0.5740429416, 0.5903154099442736, 0.6065878782040469, 0.6228603464638203, 0.6391328147235935, 0.6554052829833669]
pseudo_label_th:  0.3980698068588894
max_moda: 20.4%, max_modp: 61.4%, max_precision: 98.0%, max_recall: 20.8%, epoch: 8.0%

target_weights:  [0.0, 0.0, 0.0, 0.0, 0.9930710435442112, 0.9943342358059472, 0.9955974280676831, 0.9968606203294191, 0.9981238125911551, 0.9993870048528911]
pseudo_label_th:  0.3169209365784296
moda: 0.0%, modp: 54.3%, precision: 14.1%, recall: 86.4%
max_moda: 62.4%, max_modp: 59.4%, max_precision: 88.3%, max_recall: 72.0%, epoch: 5.0%

target_weights:  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9097386222697896, 0.9179633911828667, 0.9261881600959438, 0.934412929009021]
pseudo_label_th:  0.38797777902092656
max_moda: 69.4%, max_modp: 64.0%, max_precision: 95.3%, max_recall: 73.0%, epoch: 10.0%

target_weights:  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6081786864038533, 0.9639007223686411]
pseudo_label_th:  0.39786268125804347
max_moda: 68.2%, max_modp: 64.2%, max_precision: 97.0%, max_recall: 70.4%, epoch: 10.0%

target_weights:  [0.0, 0.0, 0.0, 0.38783079576445467, 0.45702052909058094, 0.5262102624167072, 0.5953999957428334, 0.6645897290689597, 0.733779462395086, 0.8029691957212122]
pseudo_label_th:  0.3429979306797553
moda: 0.0%, modp: 53.5%, precision: 11.7%, recall: 86.4%
max_moda: 55.4%, max_modp: 61.4%, max_precision: 85.7%, max_recall: 66.5%, epoch: 5.0%

Conclusions from above: training on target too early leads to very poor performance -> should not start earlier than epoch ~4  
setting pseudo-label threshold to close to 30 leads to very poor precision metric, but high recall.  
setting pseudo-label threshold over 40 leads to high precision but low recall.
Iäve started another run with 10 experiments, now narrowing the search interval to pseudolabel threhsold [0.37,0.42] and start epoch [4,10]

Another 10 trainings yielded 8 results at least as good as the baseline. 4 results reached ~71% max moda, and 1 result 73% max moda, which can probably be seen as a significant improvement.  I can not easily say why the poor experiments turned out as they did. In fact, the exp reaching only ~56% max moda started self-training at epoch 10, with reasonable parameters.  

target_weights:  [0.0, 0.0, 0.0, 0.0, 0.0, 0.9000313692461928, 0.9133325596859133, 0.9266337501256339, 0.9399349405653544, 0.953236131005075]  
pseudo_label_th:  0.3724288045815374  
max_moda: 73.0%, max_modp: 62.7%, max_precision: 95.0%, max_recall: 77.1%, epoch: 10.0%

target_weights:  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9097386222697896]  
pseudo_label_th:  0.39932592634030883  
max_moda: 56.5%, max_modp: 67.5%, max_precision: 98.2%, max_recall: 57.6%, epoch: 10.0%


max_moda: 64.1%, max_modp: 63.9%, max_precision: 97.8%, max_recall: 65.5%, epoch: 8.0%  
max_moda: 71.6%, max_modp: 63.4%, max_precision: 95.5%, max_recall: 75.2%, epoch: 9.0%  
max_moda: 69.0%, max_modp: 64.7%, max_precision: 93.1%, max_recall: 74.6%, epoch: 6.0%  
max_moda: 69.1%, max_modp: 61.1%, max_precision: 94.9%, max_recall: 73.0%, epoch: 10.0%  
max_moda: 70.8%, max_modp: 62.6%, max_precision: 95.8%, max_recall: 74.1%, epoch: 7.0%  
max_moda: 73.0%, max_modp: 62.7%, max_precision: 95.0%, max_recall: 77.1%, epoch: 10.0%  
max_moda: 69.5%, max_modp: 61.7%, max_precision: 95.2%, max_recall: 73.2%, epoch: 10.0%  
max_moda: 56.5%, max_modp: 67.5%, max_precision: 98.2%, max_recall: 57.6%, epoch: 10.0%  
max_moda: 71.1%, max_modp: 61.8%, max_precision: 93.0%, max_recall: 76.9%, epoch: 7.0%  
max_moda: 69.5%, max_modp: 62.9%, max_precision: 95.5%, max_recall: 73.0%, epoch: 10.0%  

mean, std  
68.42 $\pm$ 4.56

### 18/7
running 10 baseline experiments to verify whether any of the self-training results from yesterday was successful.

max_moda: 66.3%, max_modp: 65.4%, max_precision: 90.7%, max_recall: 73.8%, epoch: 5.0%  
max_moda: 66.9%, max_modp: 65.1%, max_precision: 92.1%, max_recall: 73.2%, epoch: 6.0%  
max_moda: 62.7%, max_modp: 65.3%, max_precision: 94.5%, max_recall: 66.6%, epoch: 9.0%  
max_moda: 66.9%, max_modp: 64.9%, max_precision: 93.1%, max_recall: 72.3%, epoch: 7.0%  
max_moda: 65.0%, max_modp: 66.5%, max_precision: 96.4%, max_recall: 67.5%, epoch: 10.0%  
max_moda: 71.5%, max_modp: 64.5%, max_precision: 92.3%, max_recall: 78.0%, epoch: 8.0%  
max_moda: 64.5%, max_modp: 66.7%, max_precision: 96.0%, max_recall: 67.3%, epoch: 10.0%  
max_moda: 64.5%, max_modp: 66.9%, max_precision: 94.0%, max_recall: 68.9%, epoch: 10.0%  
max_moda: 69.2%, max_modp: 65.8%, max_precision: 94.6%, max_recall: 73.4%, epoch: 10.0%  
max_moda: 66.3%, max_modp: 66.8%, max_precision: 95.0%, max_recall: 70.0%, epoch: 9.0%  

mean, std  
66.38 $\pm$ 2.39

From above UDA vs baseline experiments, the UDA yields higher max_moda in 8 of 10 experiments. However, I also note that although the seed was the same (80,81,...,89) for both sets of experiments, the training progression differs even when the target_weight is zero. For example, in the third to last training (with max_moda 56.5), the target weight is zero for all save the last epoch, but the baseline training on this seed yields much higher performance. So, the poor score of 56.5 can not be attributed to the UDA training, but rather just an unfortunate seed/run. Similarly, some runs with UDA reached max_moda even before UDA training kicked in, still beating the baseline. Another interesting observation is that while max_moda is improved for UDA, the max_modp is decreased in almost all experiments.

- [x] inspect results from yesterday to check for any qualitative difference between uda and baseline
  - [x] Note that when mvaug is used, the teacher output is aligned with the label, while the teacher ps-label is aligned with the student output. This is because mvaug is only applied to the teacher pseudo-label and the input to the student. (e.g. train_target_map_199 in 2024-07-17_12-48-57-773335/epoch_10)
  - [x] post uda training it seems like the model has got a bit better at predictive feet key points in perspective view (especially close range, which is typically missed without completely uda training). Although the pseudo-abels looks quite okay, which could possibly lead to very confident predictions post uda, the model is still not confident. Maybe it just needs more time to adapt? 20 epochs?
  - [x] it is difficult to see any obvious qualitative differences. 

- [x] start training with 20 epochs
- [x] start generalization trainings and uda trainings (5 each) on adaptation scenario 1
- [ ] check results from above trainings
  - [ ] if I'm relatively satisfied with above results, it could be a good idea to start studying GMVD benchmark
  - [ ] if I'm not satisfied, the next step is perhaps to implement cross-entropy loss and confidence weighted self-training

### 6/8

- [x] summarize results of 2,4,6->1,3,5 with/without uda and 20 epochs
  - [x] over 20 epochs, uda significantly improve moda and recall, while precision is ~unchanged, and modp slightly decreased
  - [x] while the relationship between recall and preciison metrics is clear, I don't exactly understand how moda and modp relates to each other.
      MODP def in CLEAR_MOD_HUN.py: sum(1 - distances[distances < td] / td) / np.sum(c)  
      As I understand it, modp measures the average distance of positive assignments between gt and preds? So, perhaps it is not very surprising that this metric may decrease when more positive asignments between gts and preds are achieved. I.e., when the model successfully detects more difficult pedestrians, the modp may decrease since it is more difficult to assess the exact location of these pedestrians.  
      Additionally, also in the GMVD paper, some models that achieve VERY low MODA, have competetive MODP (or even better) than models with reasonable MODA. This shows that the MODP metric alone is perhaps not very useful in a generalization/UDA setting. We can probably accept a slight decrease in modp in favor of high increase of moda.
- [x] summarize results of 2,4,5,6 -> 1,3,5,7 with/without uda and 10 epochs
  - [x] only 2 uda exps reached max_moda after uda kicked in, so it is dificult to draw conclusions
  - [x] started the same exps but with 20 epochs now
- [ ] summarize results of 2,4,5,6 -> 1,3,5,7 with/without uda and 20 epochs


**2,4,6 -> 1,3,5**

running 5 baseline exps with 20 epochs  
uda :  False  
dropview :  True  
permutation :  True  
mvaug :  True  
soft_labels :  False  
pretrained :  True  
max_moda: 65.9%, max_modp: 67.3%, max_precision: 96.7%, max_recall: 68.2%, epoch: 12.0%  
max_moda: 66.1%, max_modp: 65.0%, max_precision: 91.3%, max_recall: 73.0%, epoch: 12.0%  
max_moda: 63.7%, max_modp: 67.5%, max_precision: 95.5%, max_recall: 66.8%, epoch: 17.0%  
max_moda: 67.5%, max_modp: 66.9%, max_precision: 93.9%, max_recall: 72.3%, epoch: 8.0%  
max_moda: 61.9%, max_modp: 67.9%, max_precision: 96.4%, max_recall: 64.3%, epoch: 13.0%  

summary:
max_moda: 65.0 ± 2.0  
max_modp: 66.9 ± 1.0  
max_prec: 94.8 ± 2  
max_reca: 68.9 ± 3.3


running 5 uda exps with 20 epochs  
uda :  True  
dropview :  True  
permutation :  True  
mvaug :  True  
soft_labels :  False  
pretrained :  True  
max_moda: 74.9%, max_modp: 67.2,%, max_precision: 96.1%, max_recall: 78.0,%, epoch: 18.0% (2024-07-19_15-34-49-648126)    
max_moda: 75.8%, max_modp: 65.1,%, max_precision: 94.3%, max_recall: 80.7,%, epoch: 16.0%  
max_moda: 72.8%, max_modp: 66.3,%, max_precision: 97.1%, max_recall: 75.0,%, epoch: 15.0%  
max_moda: 72.4%, max_modp: 59.8,%, max_precision: 95.0%, max_recall: 76.4,%, epoch: 19.0%  
max_moda: 74.5%, max_modp: 60.7,%, max_precision: 95.7%, max_recall: 77.9,%, epoch: 13.0%  

summary:
max_moda: 74.1 ± 1.3  
max_modp: 63.8 ± 3.0  
max_prec: 95.6 ± 1.0  
max_reca: 77.6 ± 1.9

I conclude that training for 20 epochs without and with UDA shows significant benefits of UDA.  
moda and recall are significantly higher, while preciison is more or less the same.
However, modp decreases slightly. Why is that?

**2,4,5,6 -> 1,3,5,7** 

baseline  
uda :  False  
dropview :  True  
permutation :  True  
mvaug :  True  
soft_labels :  False  
pretrained :  True  
max_moda: 70.5%, max_modp: 66.8%, max_precision: 92.1%, max_recall: 77.1%, epoch: 7.0%  
max_moda: 69.6%, max_modp: 69.2%, max_precision: 95.9%, max_recall: 72.8%, epoch: 10.0%  
max_moda: 69.0%, max_modp: 68.4%, max_precision: 96.2%, max_recall: 71.8%, epoch: 7.0%  
max_moda: 68.6%, max_modp: 68.9%, max_precision: 97.3%, max_recall: 70.6%, epoch: 10.0%  
max_moda: 66.6%, max_modp: 67.9%, max_precision: 96.6%, max_recall: 69.0%, epoch: 8.0%  

with uda  
uda :  True  
dropview :  True  
permutation :  True  
mvaug :  True  
soft_labels :  False  
pretrained :  True  
max_moda: 74.6%, max_modp: 67.7%, max_precision: 94.6%, max_recall: 79.1%, epoch: 9.0%  
** max_moda: 72.4%, max_modp: 68.0%, max_precision: 97.0%, max_recall: 74.7%, epoch: 8.0%  
** max_moda: 73.7%, max_modp: 66.4%, max_precision: 93.2%, max_recall: 79.5%, epoch: 8.0%  
max_moda: 73.8%, max_modp: 66.0%, max_precision: 97.2%, max_recall: 76.1%, epoch: 9.0%  
** max_moda: 71.3%, max_modp: 68.8%, max_precision: 93.1%, max_recall: 77.0%, epoch: 8.0%  

** invalid because max_moda is reached before uda has kicked in.



baseline 20 epochs (GMVD report 43 moda)
max_moda: 71.5%, max_modp: 69.0%, max_precision: 95.1%, max_recall: 75.4%, epoch: 17.0%
max_moda: 70.4%, max_modp: 69.9%, max_precision: 97.3%, max_recall: 72.4%, epoch: 17.0%
max_moda: 70.2%, max_modp: 70.7%, max_precision: 97.6%, max_recall: 72.0%, epoch: 13.0%
max_moda: 70.1%, max_modp: 67.1%, max_precision: 95.5%, max_recall: 73.5%, epoch: 11.0%
max_moda: 72.5%, max_modp: 69.3%, max_precision: 96.0%, max_recall: 75.6%, epoch: 19.0%

uda 20 epochs (all valid since UDA always started no later than epoch 10)
max_moda: 78.7%, max_modp: 71.3%, max_precision: 96.1%, max_recall: 82.0%, epoch: 17.0%
max_moda: 77.7%, max_modp: 71.7%, max_precision: 96.4%, max_recall: 80.8%, epoch: 13.0%
max_moda: 79.6%, max_modp: 70.6%, max_precision: 95.8%, max_recall: 83.3%, epoch: 16.0%
max_moda: 77.3%, max_modp: 70.6%, max_precision: 96.7%, max_recall: 80.0%, epoch: 18.0%
max_moda: 78.5%, max_modp: 69.9%, max_precision: 95.9%, max_recall: 81.9%, epoch: 15.0%

Conclusion: large improvements to moda and recall, while modp and precision is relatively unchanged.


### 7/8

timeplan:

- [x] 0.5h mail & planering
- [ ] 2h analysera exps
  - [x] 1
  - [x] 2
  - [x] 3
  - [x] 4 
- [ ] 2h implementera 1,2,3,4,5,6,7 -> 2,4,5,6
  - [x] 1
  - [x] 2
  - [x] 3
  - [x] 4


3h träna med multiviewx

1h starta träningar på givna multiviewx scenarion

0.5h nedvarvning


- [x] 1,3,5 -> 2,4,6 (ONGOING)
- [x] 1,3,5,7 -> 2,4,5,6 (ONGOING)
Both above experiments resulted in UDA degrading the performance in comparison with the baseline. 
Seems like 1,3,(5),7->2,4,(5),6 is more difficult than the other way around.
I noticed that pseudo-labels seems to be of worse quality than in the successful experiments of 2,4,6->1,3,5.
I've started another set of uda experiments where uda kicks in at a later epoch. Perhaps this will give the model time enough to produce pseudo-labels of sufficient quality.
- [ ] 1,2,3,4,5,6,7 -> 1,3,5,7
- [ ] 1,2,3,4,5,6,7 -> 2,4,5,6  
For these experiments, I suspect that GMVD do not duplicate views during training, which could make the model confused when this happens during testing. It makes sense to also train the model with duplicate views.
  - [ ] implement training with duplicate views
  - [x] implement testing with duplicate views


**1,3,5 -> 2,4,6**  
BASELINE  
max_moda: 49.2%, max_modp: 57.7%, max_precision: 80.7%, max_recall: 64.6%, epoch: 17.0%  
max_moda: 53.4%, max_modp: 56.6%, max_precision: 88.1%, max_recall: 61.7%, epoch: 13.0%  
max_moda: 53.8%, max_modp: 59.0%, max_precision: 85.6%, max_recall: 64.7%, epoch: 20.0%  
max_moda: 48.5%, max_modp: 58.3%, max_precision: 86.0%, max_recall: 58.0%, epoch: 20.0%  
max_moda: 50.8%, max_modp: 56.8%, max_precision: 79.9%, max_recall: 68.0%, epoch: 10.0%  


UDA

| uda_start | weight_start | weight_end | ps-label-th | scores                                                                                 |
| --------- | ------------ | ---------- | ----------- | -------------------------------------------------------------------------------------- |
| 9         | 0.98         | 0.98       | 0.412       | max_moda: 39.2%, max_modp: 57.1%, max_precision: 83.4%, max_recall: 48.9%, epoch: 8.0% |
| 10        | 0.96         | 0.97       | 0.376       | max_moda: 42.2%, max_modp: 56.6%, max_precision: 83.1%, max_recall: 53.0%, epoch: 8.0% |
| 5         | 0.05         | 0.61       | 0.404       | max_moda: 32.7%, max_modp: 57.3%, max_precision: 91.2%, max_recall: 36.1%, epoch: 8.0% |
| 6         | 0.89         | 0.98       | 0.389       | max_moda: 32.5%, max_modp: 55.7%, max_precision: 77.3%, max_recall: 45.9%, epoch: 5.0% |
| 4         | 0.62         | 0.83       | 0.380       | max_moda: 12.7%, max_modp: 59.1%, max_precision: 67.6%, max_recall: 24.4%, epoch: 5.0% |


Note: In all but the third experiment, the model seems to produce many false positives after UDA training, resulting in low precision and moda reaching zero.
In the third experiment, however, precision goes to 100% while recall becomes very low. Could it be because the pseudo-label threshold is very high relative to the low target_epoch_start. I.e., only very select few pseudo-labels are created in beginning of uda training, leading to overfitting to a small number of pedestrians.

starting new experiments with later uda start

| uda_start | weight_start | weight_end | ps-label-th | scores                                                                                  |
| --------- | ------------ | ---------- | ----------- | --------------------------------------------------------------------------------------- |
| 14        | 0.98         | 0.98       | 0.412       | max_moda: 48.9%, max_modp: 57.0%, max_precision: 77.3%, max_recall: 69.3%, epoch: 14.0% |
| 15        | 0.96         | 0.97       | 0.376       | max_moda: 46.6%, max_modp: 57.0%, max_precision: 84.5%, max_recall: 57.1%, epoch: 12.0% |
| 10        | 0.05         | 0.61       | 0.404       | max_moda: 52.4%, max_modp: 56.5%, max_precision: 85.7%, max_recall: 62.9%, epoch: 10.0% |
| 11        | 0.89         | 0.98       | 0.389       | max_moda: 51.6%, max_modp: 56.8%, max_precision: 85.2%, max_recall: 62.4%, epoch: 9.0%  |
| 9         | 0.62         | 0.83       | 0.380       | max_moda: 44.5%, max_modp: 58.2%, max_precision: 89.7%, max_recall: 50.3%, epoch: 9.0%  |

In all cases above, the precision becomes very low and 0.0 moda is reached. => seems like a lot of false positives.
Seems like pseudo-label-th must be significantly higher for this benchmark, perhaps ~0.42


**1,3,5,7 -> 2,4,5,6**  
BASELINE (GMVD report ~28 moda)  
max_moda: 63.7%, max_modp: 66.6%, max_precision: 95.8%, max_recall: 66.6%, epoch: 7.0%  
max_moda: 69.5%, max_modp: 62.5%, max_precision: 92.4%, max_recall: 75.7%, epoch: 10.0%  
max_moda: 68.4%, max_modp: 64.4%, max_precision: 87.1%, max_recall: 80.3%, epoch: 13.0%  
max_moda: 64.8%, max_modp: 64.4%, max_precision: 91.9%, max_recall: 71.1%, epoch: 7.0%  
max_moda: 68.5%, max_modp: 66.0%, max_precision: 90.3%, max_recall: 76.7%, epoch: 19.0%  

baseline moda: 67.0 ± 2.3


UDA

| uda_start | weight_start | weight_end | ps-label-th | scores                                                                                  |
| --------- | ------------ | ---------- | ----------- | --------------------------------------------------------------------------------------- |
| 6         | 0.3          | 0.6        | 0.397       | max_moda: 57.8%, max_modp: 66.6%, max_precision: 95.4%, max_recall: 60.7%, epoch: 16.0% |
| 9         | 0.6          | 1.0        | 0.395       | max_moda: 67.4%, max_modp: 64.5%, max_precision: 88.1%, max_recall: 77.9%, epoch: 12.0% |
| 10        | 0.8          | 0.8        | 0.379       | max_moda: 71.8%, max_modp: 63.9%, max_precision: 91.2%, max_recall: 79.5%, epoch: 19.0% |
| 4         | 0.1          | 0.3        | 0.385       | max_moda: 52.2%, max_modp: 65.1%, max_precision: 90.9%, max_recall: 58.0%, epoch: 10.0% |
| 6         | 0.17         | 0.92       | 0.373       | max_moda: 61.8%, max_modp: 56.2%, max_precision: 88.2%, max_recall: 71.3%, epoch: 17.0% |



Starting a second set of experiments, now starting UDA training later since it seemed to start too early before.

UDA with later start

| uda_start | weight_start | weight_end | ps-label-th | scores                                                                                  |
| --------- | ------------ | ---------- | ----------- | --------------------------------------------------------------------------------------- |
| 11        | 0.3          | 0.6        | 0.397       | max_moda: 64.3%, max_modp: 62.6%, max_precision: 95.0%, max_recall: 67.9%, epoch: 16.0% |
| 14        | 0.6          | 1.0        | 0.395       | max_moda: 74.1%, max_modp: 62.3%, max_precision: 92.8%, max_recall: 80.3%, epoch: 15.0% |
| 15        | 0.8          | 0.8        | 0.379       | max_moda: 75.6%, max_modp: 62.1%, max_precision: 93.8%, max_recall: 81.0%, epoch: 20.0% |
| 9         | 0.1          | 0.3        | 0.385       | max_moda: 67.4%, max_modp: 63.5%, max_precision: 88.1%, max_recall: 77.9%, epoch: 10.0% |
| 11        | 0.17         | 0.92       | 0.373       | max_moda: 73.9%, max_modp: 62.6%, max_precision: 92.2%, max_recall: 80.8%, epoch: 17.0% |


uda moda: 71.0 ± 4.3

repeating the first and fourth experiment above but now with uda_start=14, ps-label-th=0.38, to see if I can pump those numbers up.


**1,2,3,4,5,6,7 -> four cameras**  

Default MVDet results:

/mimer/NOBACKUP/groups/naiss2023-23-214/mvdet/results/logs/wildtrack_frame/default/2024-07-01_18-07-44
slurm-2464253_3  
evaluated on all 7 cameras yield  
moda: 87.4%, modp: 75.5%, precision: 93.2%, recall: 94.2%

evaluated on 2,4,5,6 yields   
moda: 32.8%, modp: 65.9%, precision: 90.1%, recall: 36.8%

evaluated on 1,3,5,7 yields  
moda: 48.4%, modp: 72.3%, precision: 94.2%, recall: 51.6%

I note that, due to chance, the views sometimes end up in the ''correct'' positions when duplication is used. Therefore, the performance varies drastically between samples. Very poor performance is achieved when few views are in the expected position, while decent performance is achived when many views are in the expected position.
In reality, it doesn't make sense to ''shuffle'' the views just because some cameras are broken. It would make more sense to keep all available cameras in their original position, and put duplicates where in place of the broken cameras.

| model         | 1,2,3,4,5,6,7 | 1,3,5,7               | 2,4,5,6     |
| ------------- | ------------- | --------------------- | ----------- |
| default MVDet | moda: 87.4%   | moda: 48.4%,          | moda: 32.8% |
| General MVDet |               | (epoch 9) moda: 71.8% |             |
| UDA MVDet     |               |                       |             |


Conclusion: Seems like my training tricks does the charm also for this benchmark. GMVD has been very lazy...


**TODO**
- [ ] 7 -> 4 cameras adaptation
  - [ ] evaluate default MVDet and general MVDet in the setting where all available cameras are in the correct place. It doesn't make sense to shuffle them
- [ ] 


