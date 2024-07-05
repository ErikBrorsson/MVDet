# commands
docker run -it --gpus all --shm-size=8g -v $PWD:/code/ -v /home/gpss1/remote/datasets/Wildtrack_dataset:/data/Wildtrack -w /code mvdet

docker run -it --gpus all --shm-size=8g -v $PWD:/code/ -v /home/gpss1/remote/mnts/mnt0:/mnt -v /home/gpss1/remote/datasets/Wildtrack_dataset:/data/Wildtrack -w /code mvdet

python main.py -d wildtrack

python main.py -d wildtrack --cam_adapt --train_viz --resume 2024-06-26_11-16-08


rsync -r erikbro@alvis1:/mimer/NOBACKUP/groups/naiss2023-23-214/mvdet/results/logs/wildtrack_frame/default mnt0/


# UDA baseline

## TODO

## General
- [ ] Plot perspective view foot/head predictions on target data in all cameras (not just one as is done now)
- [ ] Investigate what component leads to poor generalization (perspective view feature extraction or BEV detection head)
  - [ ] Do NMS on perspective view predictions
  - [ ] Transform perspective view predictins to 3D
  - [ ] on target domain: compare bev detections with transformed perspective view detections 

### implement EMA teacher

- [ ] EMA teacher
  - [x] implement
  - [ ] test ema teacher (train only on supervised and simply keep an EMA teacher on the side. Then test the EMA teacher after training's finished.)





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
- [x] dropview
- [ ] 3DROM
- [ ] MVAug

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


# Notes
- Camera C3 is not undistorted properly. Perhaps they use another cameramodel for this camera? The projection of points looks alrgiht, although lines does not appear straight in this camera.
- Isn't it strange to evaluate 2,4,5,6->1,3,5,7 since camera 5 is avialable (and in the same ordering) in both camera rigs? Perhaps we are basically just evaluating the models "single camera" performance, using only camera 5, while cameras 1,3,7 are useless.

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

These predictions are competitive with the standard BEV predictions in the camp adaptation setting. However, they are not better, so it is not clear whether the domain gap lies mainly in the perspective view backbone or in the bev decode head.  
There are a few problems with the evaluation above.  
First, since the perspective view is trained to detect feet (rather than pedestrians), it cannot detect any pedestrian that is too close to the camera.  
Second, the above scheme tests also the perspective view to bev projection and sensor fusion algorithm. It is not a precise evaluation of the perspective view prediction quality.  
It would perhaps be better to evaluate the adaptation capabilities of the perspective view backbone in the perspective view.
Alt 1. evaluate perspective view detections per camera.  
Alt 2. evaluate the join perspective view detections (for example, are there any pedestrian that is missed in all cameras?)  

When evaluating different camera setups, it would be helpful to plot the field of views of each camera in the bev map to understand which parts of the bev we can expect detections in.  

After training on "2,4,5,6" the output on cam2 looks like below.  
It can be seen that the network doesn't provide very confident predictions in perspective view even on the cameras that are included in the train set.
![alt text](resources/images/output_cam2_foot_38.jpg)

When evaluating on "1,3,5,7", the predictions in camera 3 looks like below.  
The quality seems to be similar to the predictions in the training set.  
To confirm this quantitatively, maybe I should just print the loss? Rather than printing MODA/MODP, as this would require me to set thresholds and do NMS.
![alt text](resources/images/output_cam3_foot_28.jpg)

