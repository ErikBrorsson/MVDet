# Abstract
We consider the problem of UDA for MV pedestrian detection.
Our paper constitutes the first extensive study on mean teacher self-training for this problem.
We make the following contributions:
- adopt the mean teacher self-training framwork to MV pedestrian detection.
- provide strong UDA results for multiple benchmarks, constituting a stong baseline for future works to compare with.
- provide insights into how pseudo-labels should be created
- ablate the components of the framework 


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

| description                   | ema weights | persp. supervision | dropview | mvaug | pretrained | MODA | MODA EMA | varying threshold |
| ----------------------------- | ----------- | ------------------ | -------- | ----- | ---------- | ---- | -------- | ----------------- |
| baseline                      |             |                    |          |       |            |      |          |                   |
| baseline pre                  |             |                    |          |       | x          |      |          |                   |
| baseline pre w dropview       |             |                    | x        |       | x          |      |          |                   |
| baseline pre w mvaug          |             |                    |          | x     | x          |      |          |                   |
| baseline pre w d.view + mvaug |             |                    | x        | x     | x          |      |          |                   |
| baseline pre w persp.         |             | x                  |          |       | x          | ?    |          |                   |
| baseline pre w persp. + mvaug |             | x                  |          | x     | x          | ?    |          |                   |
| baseline pre w persp. and aug |             | x                  | x        | x     | x          | ?    |          |                   |
| baseline pre w. ema weights   | x           | x                  | x        | x     | x          | ?    |          |                   |


## uda method development
I would like to motivate the choices for the UDA method used:
- ema teacher (or simple let student do the labeling?)
- weak-strong aug (mvaug)
- uda persp. sup
- ps-label-trick

GMVD -> MultiviewX
| description                 | ema teacher | uda persp. | weak-strong mvaug | ps-label-trick | MODA |
| --------------------------- | ----------- | ---------- | ----------------- | -------------- | ---- |
| naive self-training         |             |            |                   | x              |      |
| uda w. ema teacher          | x           |            |                   | x              | ?    |
| uda w ema and mvaug         | x           |            | x                 | x              |      |
| uda w ema and persp.        | x           | x          |                   | x              | ?    |
| full uda                    | x           | x          | x                 | x              | ?    |
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



## Ablations
Since my pseudo-label trick only works for some datasets, while standard pseudo-labeling is better on other datasets,
I should show performance of both methods on all benchmarks. Otherwise, the user might question the necessity of my pseudo-label trick (perhaps it only works on 1 dataset?).

For the other things: mean teacher, mvaug, persp sup, uda persp sup: it would be good to test these components on a single benchmark and then apply the most successful combination on all experiments.

- no mean teacher (make pseudo-labels with student)
- no self-training (use mean-teacher and potential extra training rounds to verify that this alone does not boost performance)
- no pseudo-label tricks (i.e. use best cls_thres from pre-training to do the pseudo-labels instead of adjusting nms_thres)
- no augmentation (need to fix mvaug before this?)
- with persp. sup

| description             | mean teacher | self-training | ps-label trick | weak-strong aug | uda persp. sup | MODA |
| ----------------------- | ------------ | ------------- | -------------- | --------------- | -------------- | ---- |
| baseline                |              |               |                |                 |                |      |
| full uda                | x            | x             | x              | x               |                | ?    |
| full uda  + persp. sup  | x            | x             | x              | x               | x              | ?    |
| uda w/o ps-label trick  | x            | x             |                | x               |                | ?    |
| uda w/o weak-strong aug | x            | x             | x              |                 |                | ?    |
| uda w/o mean-teacher    |              | x             | x              | x               |                | ?    |

