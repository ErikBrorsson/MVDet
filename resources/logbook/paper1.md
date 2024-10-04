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

**Note: it doesnt really make sense that I didnt use dv+3drom for source data in method 2.**
1. Baseline: pre+dv+3drom, UDA: augmentation=[dropview, 3drom], no persp sup, max_pseudo_th=7, lambda=1.0, epochs=20
2. Baseline: pre+dv, UDA: augmentation=dropview, no persp sup, max_pseudo_th=7, lambda=1.0, epochs=20
| benchmark               | baseline | training method | ps-label-th | alpha=0   | alpha = 0.9 | alpha = 0.99 | **alpha = 0.999** | alpha = 1 |
| ----------------------- | -------- | --------------- | ----------- | --------- | ----------- | ------------ | ----------------- | --------- |
| gmvd s1c1 -> multiviewx | 70.3     | 2               | 0.3         | 86.4 done | 87.8        | 87.9         | 86.5              | 78.4      |
| multiviewx -> wildtrack | 70.0     | 2               | 0.4         | -  done   | -   done    | 79.0         | 80.9              | 79.0      |
| wildtrack -> multiviewx | 35.9     | 1               | 0.2         | 78.3      | 79.5        | 78.9         | 83.7              | 65.1      |
2902043_x
2902038_x
2901886_x

## max_pseudo_th
1. Baseline: pre+dv, UDA: augmentation=dropview, no persp sup, alpha_teacher=0.999, lambda=1.0, epochs=20
| benchmark               | baseline | training method | k_size=3 | k_size = 5 | k_size = 7 | k_size=11 | k_size=15 |
| ----------------------- | -------- | --------------- | -------- | ---------- | ---------- | --------- | --------- |
| gmvd s1c1 -> multiviewx | 70.3     | 1               | 87.3     | 87.2       | 87.5       | 87.9      | 85.6      |
| multiviewx -> wildtrack | 70.0     | 1               | 82.6     | 82.1       | 81.4       | 78.9      | 67.9      |
2895196 _ x




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
| benchmark                    | baseline         | jobscript           | ps-label-th | base uda** | base+dv | base+mv | base+3dr | base + persp | full uda |
| ---------------------------- | ---------------- | ------------------- | ----------- | ---------- | ------- | ------- | -------- | ------------ | -------- |
| multiviewx -> wildtrack      | 70.0 2883235_356 | 490-494 **ONGOING** | 0.4         |            |         |         |          |              |          |
| wildtrack -> multiviewx      | 35.9 2883159_416 | 500-504 **ONGOING** | 0.2         |            |         |         |          |              |          |
| wildtrack 2,4,5,6 -> 1,3,5,7 | 75.2 2883159_436 | 520-524             |             |            |         |         |          |              |          |
| wildtrack 1,3,5,7 -> 2,4,5,6 | 72.3 2883159_426 | 530-534             |             |            |         |         |          |              |          |
| multiviewx cam adapt         | 54.7 2883235_446 | 540-544             |             |            |         |         |          |              |          |
| gmvd s1c1 -> multiviewx      | 70.3 2861528_392 | 470-474             |             |            |         |         |          |              |          |
| gmvd s1c2 -> multiviewx      | 66.9 2883159_406 | 480-484             |             |            |         |         |          |              |          |
**with tuned ps-label-strat and ema. The baseline data aug is applied to source data, while the different augmentation methods here refers to strong-weak self-training aug. 



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



