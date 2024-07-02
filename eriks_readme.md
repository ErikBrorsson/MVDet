# commands
docker run -it --gpus all --shm-size=8g -v $PWD:/code/ -v /home/gpss1/remote/datasets/Wildtrack_dataset:/data/Wildtrack -w /code mvdet


python main.py -d wildtrack

python main.py -d wildtrack --cam_adapt --train_viz --resume 2024-06-26_11-16-08


# UDA baseline

## TODO


### implement EMA teacher

- [ ] implement the EMA teacher





### create pseudo labels
- [x] train with soft labels
- [ ] train with confidence weighted MSE loss
  - [ ] create confidence scores from model prediction which is in range -infty to +infty
- [ ] create pseudo-labels in bev and perspective view separately
  - [x] find probabilities and argmax
  - [ ] non-maximum supression
  - [ ] perspective view 
- [ ] train with pseudo-labels
- [x] create pseudo-labels in bev and project into perspective view
  - [x] find the pos of pseudo-labels in bev
  - [x] project pos to cameras
  - [ ] plot pseudo-labels during training

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


There should probably be more focus on accurate source labels in the beginning, and then successively focus is shifted to target domain as the quality of the pseudo-labels increase.


# Experiments

### all cameras MVDet baseline
/mimer/NOBACKUP/groups/naiss2023-23-214/mvdet/results/logs/wildtrack_frame/default/2024-07-01_18-07-44
slurm-2464253_3
moda: 87.4%, modp: 75.5%, precision: 93.2%, recall: 94.2%

### cam_adapt 1,3,5,7 -> 2,4,5,6


### cam_adapt 2,4,5,6 -> 1,3,5,7



# Notes
- Camera C3 is not undistorted properly. Perhaps they use another cameramodel for this camera? The projection of points looks alrgiht, although lines does not appear straight in this camera.

# log book
### 1/7
started baseline experiments

### 2/7
The experiemnts yesterday didn't turn out well. The model cannot generalize. From the appearance of the predictions, it seems like the wrong calibration matrices are used.  
Indeed, it seems like the code is incorrect.  Iä've fixed this and started new experiments on cam_adaptation.  





