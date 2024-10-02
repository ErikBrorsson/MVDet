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
| benchmark                    | uda naive                          | uda max_pseudo                     |
| ---------------------------- | ---------------------------------- | ---------------------------------- |
| multiviewx -> wildtrack      | x (th=0.2), x (th=0.3), x (th=0.4) | x (th=0.2), x (th=0.3), x (th=0.4) |
| wildtrack -> multiviewx      | x (th=0.2), x (th=0.3), x (th=0.4) | x (th=0.2), x (th=0.3), x (th=0.4) |
| wildtrack 2,4,5,6 -> 1,3,5,7 | x (th=0.2), x (th=0.3), x (th=0.4) | x (th=0.2), x (th=0.3), x (th=0.4) |
| wildtrack 1,3,5,7 -> 2,4,5,6 | x (th=0.2), x (th=0.3), x (th=0.4) | x (th=0.2), x (th=0.3), x (th=0.4) |
| multiviewx cam adapt         | x (th=0.2), x (th=0.3), x (th=0.4) | x (th=0.2), x (th=0.3), x (th=0.4) |
| gmvd s1c1 -> multiviewx      | x (th=0.2), x (th=0.3), x (th=0.4) | x (th=0.2), x (th=0.3), x (th=0.4) |
| gmvd s1c2 -> multiviewx      | x (th=0.2), x (th=0.3), x (th=0.4) | x (th=0.2), x (th=0.3), x (th=0.4) |

Figure 1: MODA over time (over training epochs), comparison between uda naive and uda max_pseudo, to see difference in stability.

Table 6: with different data augmentation performance
Bring in some of the supplementary material here, e.g. augmentation for baseline and/or augmentation for uda.


# Further experiments (probably appendix)
## Baseline development
I would like to motivate the choices for the baseline.
- ema weights (boosts performance even without UDA?)
- persp. supervision?
- mvaug?
- varying threshold?
- ext architecture?

Here I motivate which of the above I use for the baseline on ALL datasets. 
Perhaps it is reasonable to use 2 benchmarks for developing the baseline?
E.g. gmvd s1c1 -> multiviewx and multiviewx -> wildtrack (one camera rig adaptation and one sim2real adaptation)
It is reasonable to choose a method that yields best results with a varying threshold, since this threshold can easily be selected by manual/visual inspection.
Show how much worse the baseline results is without varying threshold.

GMVD s1c1 -> MultiviewX
| description                       | ema weights | persp. supervision | dropview | mvaug | pretrained | MODA             | MODA EMA | varying threshold |
| --------------------------------- | ----------- | ------------------ | -------- | ----- | ---------- | ---------------- | -------- | ----------------- |
| baseline                          |             |                    |          |       |            | 36.9 2832981_340 |          |                   |
| baseline pre                      |             |                    |          |       | x          | 64.6 2833240_341 |          |                   |
| baseline pre w dropview           |             |                    | x        |       | x          | 65.8 2833867_342 |          |                   |
| baseline pre w mvaug              |             |                    |          | x     | x          | 66.1 2833240_343 |          |                   |
| baseline pre w d.view + mvaug     |             |                    | x        | x     | x          | 67.9 2833240_344 |          |                   |
| baseline pre w persp.             |             | x                  |          |       | x          | 66.4 2833240_345 |          |                   |
| baseline pre w persp. + mvaug     |             | x                  |          | x     | x          | 69.2 2833240_346 |          |                   |
| baseline pre w persp. + dv        |             | x                  | x        |       | x          | 66.3 2833883_347 |          |                   |
| baseline pre w persp.+ dv + mvaug |             | x                  | x        | x     | x          | 69.1 2833883_348 |          |                   |
| baseline pre w. ema weights       | x           | x                  | x        | x     | x          |                  |          |                   |

Conclusion: use pre + dv + mv + persp

MultiviewX -> Wildtrack
| description                       | ema weights | persp. supervision | dropview | mvaug | pretrained | MODA             | MODA EMA | varying threshold |
| --------------------------------- | ----------- | ------------------ | -------- | ----- | ---------- | ---------------- | -------- | ----------------- |
| baseline                          |             |                    |          |       |            | 52.5 2833966_350 |          |                   |
| baseline pre                      |             |                    |          |       | x          | 69.5 2833966_351 |          |                   |
| baseline pre w dropview           |             |                    | x        |       | x          | 72.9 2833966_352 |          |                   |
| baseline pre w mvaug              |             |                    |          | x     | x          | 69.0 2833966_353 |          |                   |
| baseline pre w d.view + mvaug     |             |                    | x        | x     | x          | 70.1 2833966_354 |          |                   |
| baseline pre w persp.             |             | x                  |          |       | x          | 70.9 2833966_355 |          |                   |
| baseline pre w persp. + mvaug     |             | x                  |          | x     | x          | 68.8 2833966_356 |          |                   |
| baseline pre w persp. + dv        |             | x                  | x        |       | x          | 73.3 2833966_357 |          |                   |
| baseline pre w persp.+ dv + mvaug |             | x                  | x        | x     | x          | 70.1 2833966_358 |          |                   |
| baseline pre w. ema weights       | x           | x                  | x        | x     | x          |                  |          |                   |

Conclusion: use pre + dv + persp (no mvaug)


Baseline development: since results differ with different datasets, it may be reasonable to run on all datasets during baseline development.
| benchmark                    | base | base+pre | base+pre+persp | base+pre+dv | base+pre+mv | base+pre+3dr | full baseline |
| ---------------------------- | ---- | -------- | -------------- | ----------- | ----------- | ------------ | ------------- |
| multiviewx -> wildtrack      |      |          |                |             |             |              |               |
| wildtrack -> multiviewx      |      |          |                |             |             |              |               |
| wildtrack 2,4,5,6 -> 1,3,5,7 |      |          |                |             |             |              |               |
| wildtrack 1,3,5,7 -> 2,4,5,6 |      |          |                |             |             |              |               |
| multiviewx cam adapt         |      |          |                |             |             |              |               |
| gmvd s1c1 -> multiviewx      |      |          |                |             |             |              |               |
| gmvd s1c2 -> multiviewx      |      |          |                |             |             |              |               |


## uda method development
I would like to motivate the choices for the UDA method used:
- ema teacher (or simple let student do the labeling?)
- weak-strong aug (mvaug)
- uda persp. sup
- ps-label-trick

GMVD -> MultiviewX
| description                 | ema teacher | uda persp. | weak-strong mvaug | ps-label-trick | MODA |
| --------------------------- | ----------- | ---------- | ----------------- | -------------- | ---- |
| naive uda                   |             |            |                   | x              |      |
| uda + ema                   | x           |            |                   | x              | ?    |
| uda + ema + aug             | x           |            | x                 | x              |      |
| uda + ema + persp.          | x           | x          |                   | x              | ?    |
| uda + ema + persp + aug     | x           | x          | x                 | x              | ?    |
| full uda w/o ps-label-trick | x           | x          | x                 |                | ?    |

Considering that the pseudo-label trick seems necessary for uda to work at all, it seems reasonable to have it activated for all but one experiment.
The last experiment shows that uda doesnät work without it.

MultiviewX -> Wildtrack
| description                 | ema teacher | uda persp. | weak-strong mvaug | ps-label-trick | MODA |
| --------------------------- | ----------- | ---------- | ----------------- | -------------- | ---- |
| naive self-training         |             |            |                   |                |      |
| uda w. ema teacher          | x           |            |                   |                | ?    |
| uda w ema and mvaug         | x           |            | x                 |                |      |
| uda w ema and persp.        | x           | x          |                   |                | ?    |
| full uda                    | x           | x          | x                 |                | ?    |
| full uda w/o ps-label-trick | x           | x          | x                 | x              | ?    |


Baseline development: since results differ with different datasets, it may be reasonable to run on all datasets during baseline development.
| benchmark                    | base uda** | base+dv | base+mv | base+3dr | base + persp | full uda |
| ---------------------------- | ---------- | ------- | ------- | -------- | ------------ | -------- |
| multiviewx -> wildtrack      |            |         |         |          |              |          |
| wildtrack -> multiviewx      |            |         |         |          |              |          |
| wildtrack 2,4,5,6 -> 1,3,5,7 |            |         |         |          |              |          |
| wildtrack 1,3,5,7 -> 2,4,5,6 |            |         |         |          |              |          |
| multiviewx cam adapt         |            |         |         |          |              |          |
| gmvd s1c1 -> multiviewx      |            |         |         |          |              |          |
| gmvd s1c2 -> multiviewx      |            |         |         |          |              |          |
**with tuned ps-label-strat (otherwise it doesnt work at all) and ema (cause why not)
Just pick the best ps-label-threshold that I've found so far for each benchmark (preferably use max-pseudo for all exps).
These experiments are used to set the uda augmentation and persp sup strategy.
After this, I may run more experiments with the specific choice of augmentation and persp sup, which may result in finding better ps-label-threshold.
But that won't make this table useless/outdated. Obviously, I cannot do a joint grid search on all parameters to find the optimal, because the search space becomes too big.
I need to set some parameters at a time. 
For all these exps, I will load weights from the developed baseline, and the chosen baseline augmentation will be applied to the source data.

## mean teacher paramter
Do experiments on one cam-adapt benchmark (gmvd->multiviewx) and one domain adaptation benchmark (multiviewx -> wildtrack)

**ONGOING**

- mean teacher alpha=0.9
- mean teacher alpha=0.99
- mean teacher alpha=0.999 (used in MIC)
- mean teacher alpha=0. (no mean teacher)

## Ls + lambda*Lt
lambda parameter weighting the target loss

Do experiments on one cam-adapt benchmark (gmvd->multiviewx) and one domain adaptation benchmark (multiviewx -> wildtrack)

constant value seems reasonable.

lambda = 0.1, 0.3, 0.5, 0.75, 1.0



