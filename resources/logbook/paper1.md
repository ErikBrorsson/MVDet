# TODO

- [x] baseline development (develop a strong generalization baseline, which is also used as the pretraining step for UDA)
- [x] pseudo-labelling method (simple self-training without data augmentation. Here I choose appropriate pseudo-label thresholds for each benchmark and choose pseudo-labelling strategy) 
- [x] uda development (After choosing ps-label-th and pseduo-labelling method above. I choose data augmentation and persp supervision. 5 epochs, 0.99 alpha teacher, 1.0 lambda, max_pseudo)
- [x] UDA sota exps (running with 0.999 alpha teacher and 20 epochs boosts performance slightly on all but one benchmark)
- [ ] alpha teacher experiments (analyze the importance of the teacher model, alpha ranging from 0 to 1. Also show how it is connected to the number of epochs.)
- [ ] max-pseudo-threshold table (show that my pseudo-labelling method is robust to the choice of max-pseudo-threshold)
- [ ] lambda table (show performance of different lambdas. It would make sense to design the loss as (1-lambda)*Ls + lambda*Lt but perhaps it is too late for that)


# Abstract
We consider the problem of UDA for MV pedestrian detection.
Our paper constitutes the first extensive study on mean teacher self-training for this problem.
We make the following contributions:
- adopt the mean teacher self-training framwork to MV pedestrian detection.
- provide strong UDA results for multiple benchmarks, constituting a stong baseline for future works to compare with.
- provide insights into how pseudo-labels should be created
- ablate the components of the framework 

# Results

On all benchmarks, present results for
1. standard pseudo-labeling: ps-label-th=(best moda from pretraining)
2. low cls_thres high nms_thres: pseudo_th=0.2 and nms_th=40 for all (will produce good results on some datasets, worse on others)
3. max_pseudo: 0.2 for all with fixed kernel


Table 1: Real-world  data camera adaptation
| benchmark        | baseline         | uda                                                                 | oracle |
| ---------------- | ---------------- | ------------------------------------------------------------------- | ------ |
| 2,4,5,6->1,3,5,7 | 70.4 2826072_320 | **70** degen. (fp stairs) 2826662_330, with mvaug slurm-2826825_330 | 81     |
| 1,3,5,7->2,4,5,6 | 65.3 2826072_321 | 77.8 ongoing 2826672_331                                            | 85     |

** ongoing with ps-label-th=0.4 and 0.3 (optimal from pretraining)
ps_label_th=0.4 => moda 77.0 slurm-2829119_330
ps_label_th=0.3 => moda 73.5 slurm-2829123_330
also training a new baseline with mvaug, which was used in the successful uda exp in eriks_readme

Table 2: simulated data camera adaptation
| benchmark                        | baseline         | uda                  | oracle |
| -------------------------------- | ---------------- | -------------------- | ------ |
| gmvd scene1 conf 1 -> multiviewx | 64.7 2826072_322 | 81.9  2826869_332    | ~90    |
| gmvd scene1 conf 2 -> multiviewx | 62.5 2826072_323 | 79.3  2826869_333    | ~90    |
| multiviewx cam adapt             | 49.9 2826575_326 | **52.9** 2826869_336 | ~70    |

** ongoing with ps-label-th=0.4 and 0.3 (optimal from pretraining)
ps_label_th=0.4 => moda 30
ps_label_th=0.3 => moda 54.1
also training a new baseline with mvaug, which was  used in the successful uda exp in eriks_readme **OK**
also repeated the successful exp from eriks_readme (startin gfrom same pretrained model) but wiht a new seed **OK**
, and ONGOING uda exp with 40 nms_th

Table 3: sim2real and real2sim adaptation
| benchmark               | baseline         | uda               | oracle |
| ----------------------- | ---------------- | ----------------- | ------ |
| multiviewx->wildtrack   | 72.8 2826072_324 | 77.5  2826869_334 | 87     |
| wildtrack -> multiviewx | 40.2 2826072_325 | 78.8  2826869_335 | 88     |

The above results show that mean-teacher self-training is a valuable UDA method for mv pedestrian detection, both when it comes to camera-rig adaptation and sim2real adaptation.

To keep it simple, it may be best to only show the best results in the above tables (i.e. only one uda method)

# Ablations
Since my pseudo-label trick only works for some datasets, while standard pseudo-labeling is better on other datasets,
I should show performance of both methods on all benchmarks. Otherwise, the user might question the necessity of my pseudo-label trick (perhaps it only works on 1 dataset?).

For the other things, that probably seems a bit more general (should work on all datasets): mean teacher, mvaug, persp sup, uda persp sup: it would be good to test these components on a single benchmark and then apply the most successful combination on all experiments.

- no mean teacher (make pseudo-labels with student)
- no self-training (use mean-teacher and potential extra training rounds to verify that this alone does not boost performance)
- no pseudo-label tricks (i.e. use best cls_thres from pre-training to do the pseudo-labels instead of adjusting nms_thres)
- no augmentation (need to fix mvaug before this?)
- with persp. sup

Table 4: Ablation study of UDA components
| description             | mean teacher | self-training | ps-label trick | weak-strong aug | uda persp. sup | MODA |
| ----------------------- | ------------ | ------------- | -------------- | --------------- | -------------- | ---- |
| baseline                |              |               |                |                 |                |      |
| full uda                | x            | x             | x              | x               | x              | ?    |
| uda w/o persp. sup      | x            | x             | x              | x               |                | ?    |
| uda w/o weak-strong aug | x            | x             | x              |                 | x              | ?    |
| uda w/o mean-teacher    |              | x             | x              | x               | x              | ?    |
| uda w/o ps-label trick  | x            | x             |                | x               | x              | ?    |

# Analysis of extra interesting/important components

Table 5: Naive pseudo-labelling vs max_pseudo-labelling
Baseline: pre+dv+3drom (no mvaug, no persp sup)
UDA: no augmentation, no persp sup, max_pseudo_th=7, lambda=1.0, alpha_teacher=0.99, epochs=5
| benchmark                    | baseline         | uda naive                                   | uda max_pseudo                              |
| ---------------------------- | ---------------- | ------------------------------------------- | ------------------------------------------- |
| multiviewx -> wildtrack      | 70.0 2883235_356 | 19.9 (th=0.2), 42.5 (th=0.3), 78.6 (th=0.4) | 55.8 (th=0.2), 70.8 (th=0.3), 75.8 (th=0.4) |
| wildtrack -> multiviewx      | 35.9 2883159_416 | 0.0 (th=0.2), 48.1 (th=0.3), 47.9 (th=0.4)  | 73.2 (th=0.2), 68.7 (th=0.3), 43.5 (th=0.4) |
| wildtrack 2,4,5,6 -> 1,3,5,7 | 75.2 2883159_436 | 0 (th=0.2), 73.8 (th=0.3), 78.5 (th=0.4)    | 65.3 (th=0.2), 78.6 (th=0.3), 77.7 (th=0.4) |
| wildtrack 1,3,5,7 -> 2,4,5,6 | 72.3 2883159_426 | 0.4 (th=0.2), 57.9 (th=0.3), 73.4 (th=0.4)  | 71.0 (th=0.2), 79.8 (th=0.3), 60.6 (th=0.4) |
| multiviewx cam adapt         | 54.7 2883235_446 | 15.5 (th=0.2), 40.6 (th=0.3), 55.2 (th=0.4) | 58.1 (th=0.2), 63.1 (th=0.3), 56.3 (th=0.4) |
| gmvd s1c1 -> multiviewx      | 70.3 2861528_392 | 69.1 (th=0.2), 87.8 (th=0.3), 81.5 (th=0.4) | 73.4 (th=0.2), 87.8 (th=0.3), 81.3 (th=0.4) |
| gmvd s1c2 -> multiviewx      | 66.9 2883159_406 | 0 (th=0.2), 74.9 (th=0.3), 82.8 (th=0.4)    | 79.9 (th=0.2), 88.1 (th=0.3), 80.1 (th=0.4) |
2903285_x
2903311_x
2902658_x

Figure 1: comparison pseudo-labels between naive pseudo-labelling and max_pseudo-labelling.
This will help me explain why we introduce max_pseudo. 

Figure 2: MODA over time (over training epochs), comparison between uda naive and uda max_pseudo, to see difference in stability.

Table 6: with different data augmentation performance
Bring in some of the supplementary material here, e.g. augmentation for baseline and/or augmentation for uda.


**Table 7**: Mean teacher alpha parameter. alpha=0 may yield instability, while alpha=1 doesn't allow for improving pseudo-labels over time.
Show 2 benchmarks and have one row for epochs=5 and one for epochs=20. This shows that 0.99 is reasonable for epochs=5, while 0.999 may be beneficial for longer trainings.
Since the number of epochs should probably be increased when data augmentation is used, it makes sense to make these runs with the full UDA method (using augmentation).
**ONGOING**: runs on gmvds1c1 and mvx->wildtrack, 5 epochs. Baseline=pre+dv+3drom, uda=dv+3drom (max-pseudo=7, alpha_teacher=0.99, lambda=1.0, epochs=5)


**Table 8**: max_pseudo_th, shwoing robustness to varying max-pseudo-th.
**TODO**: exps on mvx -> wildtrack and gmvds1c1. Same as table 7, but varying max-pseudo-th instead. alpha_teacher=0.99 and 5 epochs should do.




# Further experiments (probably appendix)
## Baseline development
I would like to motivate the choices for the baseline.
- ema weights (boosts performance even without UDA?)
- persp. supervision?
- mvaug?
- varying threshold?
- ext architecture?

Baseline development: since results differ with different datasets, it may be reasonable to run on all datasets during baseline development.
| benchmark                    | base             | base+pre         | base+pre+persp    | base+pre+dv      | base+pre+mv      | base+pre+3dr     | full             | full - mv        |
| ---------------------------- | ---------------- | ---------------- | ----------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| multiviewx -> wildtrack      | 46.3 2847441_350 | 72.4 2847441_351 | 72.2 2847441_355  | 73.2 2847441_352 | 67.1 2846215_353 | 70.4 2858672_359 | 67.8 2883235_357 | 70.0 2883235_356 |
| wildtrack -> multiviewx      | 16.9 2870312_410 | 32.0             | 33.3              | 35.0             | 30.1             | 36.1             | 32.1 2883159_417 | 35.9 2883159_416 |
| wildtrack 2,4,5,6 -> 1,3,5,7 | 64.9 2872132_430 | 68.7             | 68.9              | 70.0             | 71.3             | 74.6             | 72.3 2883159_437 | 75.2 2883159_436 |
| wildtrack 1,3,5,7 -> 2,4,5,6 | 46.6 2872126_420 | 62.1             | 56.9              | 65.5             | 59.6             | 66.2             | 66.8 2883159_427 | 72.3 2883159_426 |
| multiviewx cam adapt         | 28.1 2878318_440 | 46.2             | 47.7              | 51.2             | 52.5             | 52.5             | 53.7 2883235_447 | 54.7 2883235_446 |
| gmvd s1c1 -> multiviewx      | 35.3 2847447_340 | 60.5 2847447_341 | 66.0  2847447_345 | 65.1 2847447_342 | 64.3 2846137_343 | 70.8 2861528_391 | 70.7 2861528_394 | 70.3 2861528_392 |
| gmvd s1c2 -> multiviewx      | 35.1 2870292_400 | 60.0             | 60.3              | 57.6             | 65.4             | 64.7             | 68.4 2883159_407 | 66.9 2883159_406 |
number of experiments in which each strategy yielded a significant performance boost/decrease:
pretraining: 7/7, 0/7
persp: 1/7, 1/7
dv: 6/7, 1/7
mv: 4/7, 3/7
3dr: 6/7, 1/7


## uda method development
I would like to motivate the choices for the UDA method used:
- ema teacher (or simple let student do the labeling?)
- weak-strong aug (mvaug)
- uda persp. sup
- ps-label-trick

Baseline: pre+dv+3drom (no mvaug, no persp sup)
UDA: max_pseudo_th=7, lambda=1.0, alpha_teacher=0.99, epochs=5
| benchmark                    | baseline         | jobscript | ps-label-th | base uda** | base+dv | base+mv | base+3dr | base + persp | base+dv+mv+3drom | base+dv+3drom |
| ---------------------------- | ---------------- | --------- | ----------- | ---------- | ------- | ------- | -------- | ------------ | ---------------- | ------------- |
| multiviewx -> wildtrack      | 70.0 2883235_356 | 490-494   | 0.4         | 76.8       | 79.7    | 80.8    | 85.0     | 75.5         | 81.8             | 84.7          |
| wildtrack -> multiviewx      | 35.9 2883159_416 | 500-504   | 0.2         | 73.1       | 77.4    | 76.0    | 79.8     | 72.8         | 80.7             | 82.4          |
| wildtrack 2,4,5,6 -> 1,3,5,7 | 75.2 2883159_436 | 520-524   | 0.3         | 78.0       | 79.3    | 79.4    | 79.2     | 78.2         | 79.0             | 79.4          |
| wildtrack 1,3,5,7 -> 2,4,5,6 | 72.3 2883159_426 | 530-534   | 0.3         | 79.9       | 81.9    | 80.6    | 79.5     | 79.9         | 80.0             | 81.4          |
| multiviewx cam adapt         | 54.7 2883235_446 | 540-544   | 0.3         | 62.9       | 63.6    | 65.1    | 63.3     | 62.8         | 62.6             | 64.2          |
| gmvd s1c1 -> multiviewx      | 70.3 2861528_392 | 470-474   | 0.3         | 88.0       | 88.3    | 87.1    | 88.8     | 87.3         | 87.0             | 89.0          |
| gmvd s1c2 -> multiviewx      | 66.9 2883159_406 | 480-484   | 0.3         | 87.9       | 87.8    | 87.7    | 89.1     | 87.8         | 87.4             | 88.8          |
**with tuned ps-label-strat and ema. The baseline data aug is applied to source data, while the different augmentation methods here refers to strong-weak self-training aug. 
number of experiments in which each strategy yielded a significant performance boost/decrease:
persp: 0/7, 0/7
dv: 5/7, 0/7
mv: 5/7, 1/7
3dr: 6/7, 0/7
=> use dv+mv+3drom 

Try ablating mv aug just since it wasnt used in baseline (and because it's complicated) => mv aug degrades performance. 
**use base+dv+3DROM**

jobscripts 630-636
slurm-2905426_63x
| benchmark                    | baseline         | ps-label-th | base+dv+3drom (20 epochs, 0.999 ema) |
| ---------------------------- | ---------------- | ----------- | ------------------------------------ |
| multiviewx -> wildtrack      | 70.0 2883235_356 | 0.4         | 82.9                                 |
| wildtrack -> multiviewx      | 35.9 2883159_416 | 0.2         | 83.6                                 |
| wildtrack 2,4,5,6 -> 1,3,5,7 | 75.2 2883159_436 | 0.3         | 79.4                                 |
| wildtrack 1,3,5,7 -> 2,4,5,6 | 72.3 2883159_426 | 0.3         | 84.9                                 |
| multiviewx cam adapt         | 54.7 2883235_446 | 0.3         | 68.9                                 |
| gmvd s1c1 -> multiviewx      | 70.3 2861528_392 | 0.3         | 89.8                                 |
| gmvd s1c2 -> multiviewx      | 66.9 2883159_406 | 0.3         | 90.2                                 |

## Ls + lambda*Lt

| benchmark                   | baseline | lambda=0.1 | lambda = 0.5                   | **lambda = 1.0** | lambda = 2.0                  | linear ramp |
| --------------------------- | -------- | ---------- | ------------------------------ | ---------------- | ----------------------------- | ----------- |
| gmvd s1c1 -> multiviewx     | 70.3     | 85.3       | 88.4                           | 87.8             | 87.8                          | 88.8        |
| multiviewx -> wildtrack_uda | 70.0     | 74.8       | 77.8 ongoing slurm-2895183_465 | 79.9             | 82.2 ongoin slurm-2895183_469 | 73.9        |

2890333_46x, 2890461_463, 2890443_467, 2890821_46x, 2895183_

For these experiments, I use 
baseline=pre+dv+3drom (from above baseline dev table)
target loss weight = [0.1, 0.5, 1.0, 2.0, linearly increasing from 0.1 to 1.0]
alpha_teacher = 0.99
max-pseudo = True
pseudo-label-th = 0.3 for gmvd and 0.4 for multiviewx->wildtrack
UDA_aug = dropview



