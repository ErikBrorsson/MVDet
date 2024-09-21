# Abstract
We consider the problem of UDA for MV pedestrian detection.
Our paper constitutes the first extensive study on mean teacher self-training for this problem.
We make the following contributions:
- adopt the mean teacher self-training framwork to MV pedestrian detection.
- provide strong UDA results for multiple benchmarks, constituting a stong baseline for future works to compare with.
- provide insights into how pseudo-labels should be created
- ablate the components of the framework 


Table 1: Real-world  data camera adaptation
| benchmark        | baseline         | uda                                                          | oracle |
| ---------------- | ---------------- | ------------------------------------------------------------ | ------ |
| 2,4,5,6->1,3,5,7 | 70.4 2826072_320 | degen. (fp stairs) 2826662_330, with mvaug slurm-2826825_330 | 81     |
| 1,3,5,7->2,4,5,6 | 65.3 2826072_321 | 77.8 ongoing 2826672_331                                     | 85     |


Table 2: simulated data camera adaptation
| benchmark                        | baseline         | uda                  | oracle |
| -------------------------------- | ---------------- | -------------------- | ------ |
| gmvd scene1 conf 1 -> multiviewx | 64.7 2826072_322 | ongoing  2826869_332 | ~90    |
| gmvd scene1 conf 2 -> multiviewx | 62.5 2826072_323 | ongoing  2826869_333 | ~90    |
| multiviewx cam adapt             | 50 2826575_326   | ongoing  2826869_336 | ~70    |


Table 3: sim2real and real2sim adaptation
| benchmark               | baseline         | uda                  | oracle |
| ----------------------- | ---------------- | -------------------- | ------ |
| multiviewx->wildtrack   | 72.8 2826072_324 | ongoing  2826869_334 | 87     |
| wildtrack -> multiviewx | 40.2 2826072_325 | ongoing  2826869_335 | 88     |

The above results show that mean-teacher self-training is a valuable UDA method for mv pedestrian detection, both when it comes to camera-rig adaptation and sim2real adaptation.

## Ablations
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


## Baseline development
I would like to motivate the choices for the baseline.
How much does data augmentation and persp supervision contribute to the baseline's performance?
Does ema weights boost outside the self-training framework?

| description             | ema weights | persp. supervision | gmvd+ (min_max_mean or weighted sum) | augmentation | MODA |
| ----------------------- | ----------- | ------------------ | ------------------------------------ | ------------ | ---- |
| baseline                |             |                    | x                                    | x            |      |
| baseline w persp. sup   |             | x                  | x                                    | x            | ?    |
| baseline w. ema weights | x           | x                  | x                                    | x            | ?    |

