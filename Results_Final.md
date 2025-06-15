List of experiments:
- Euclidean                             [DINOv2]
- Lorentz train curvature   SGD6        [DINOv2]
- Lorentz frozen curvature  SGD4        [DINOv2]
- Lorentz with HCD setup    SGD-HCD     [DINOv2]
- Poincare with HCD setup   HCD         [DINOv2]
- Poincare HCD angel only   HCD-angle   


# DINO

| Dataset  | Hyp_Kmeans | Acc_All | Acc_Old | Acc_New | wandb                                                                  | Experiment                 |
| -------- | ---------- | ------- | ------- | ------- | ---------------------------------------------------------------------- | -------------------------- |
| Aircraft | False      | 42.210% | 53.660% | 30.750% | [8fpqckq6](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/8fpqckq6) | Euclidean                  |
| CUB      | False      | 47.290% | 61.890% | 34.620% | [cb93l49j](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/cb93l49j) | Euclidean                  |
| SCars    | False      | 44.700% | 67.020% | 23.160% | [ce4ultl0](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/ce4ultl0) | Euclidean                  |
| Aircraft | True       | 55.090% | 62.110% | 48.050% | [gkx5u4iz](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/gkx5u4iz) | Lorentz HCD Setup          |
| Aircraft | False      | 54.130% | 60.850% | 47.390% | [gkx5u4iz](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/gkx5u4iz) | Lorentz HCD Setup          |
| CUB      | True       | 60.180% | 75.490% | 45.020% | [979w86nu](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/979w86nu) | Lorentz HCD Setup          |
| CUB      | False      | 56.700% | 71.640% | 41.890% | [979w86nu](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/979w86nu) | Lorentz HCD Setup          |
| SCars    | True       | 55.570% | 77.530% | 34.380% | [jlve0elr](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jlve0elr) | Lorentz HCD Setup          |
| SCars    | False      | 52.440% | 76.060% | 29.660% | [jlve0elr](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jlve0elr) | Lorentz HCD Setup          |
| Aircraft | True       | 52.570% | 62.590% | 42.520% | [ob4xrtq1](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/ob4xrtq1) | Lorentz Frozen Curvature   |
| Aircraft | False      | 53.880% | 59.050% | 47.090% | [ob4xrtq1](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/ob4xrtq1) | Lorentz Frozen Curvature   |
| CUB      | True       | 53.660% | 71.010% | 36.460% | [3sjepj1t](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/3sjepj1t) | Lorentz Frozen Curvature   |
| CUB      | False      | 59.410% | 70.870% | 48.040% | [3sjepj1t](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/3sjepj1t) | Lorentz Frozen Curvature   |
| SCars    | True       | 48.790% | 67.930% | 30.320% | [teuxctd3](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/teuxctd3) | Lorentz Frozen Curvature   |
| SCars    | False      | 49.090% | 67.170% | 31.640% | [teuxctd3](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/teuxctd3) | Lorentz Frozen Curvature   |
| Aircraft | True       | 48.540% | 57.670% | 39.400% | [ob4xrtq1](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/ob4xrtq1) | Lorentz Train Curvature    |
| Aircraft | False      | 47.430% | 53.900% | 40.960% | [ob4xrtq1](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/ob4xrtq1) | Lorentz Train Curvature    |
| CUB      | True       | 48.650% | 64.770% | 32.680% | [3sjepj1t](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/3sjepj1t) | Lorentz Train Curvature    |
| CUB      | False      | 57.900% | 71.780% | 44.160% | [3sjepj1t](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/3sjepj1t) | Lorentz Train Curvature    |
| SCars    | True       | 54.470% | 75.030% | 34.640% | [teuxctd3](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/teuxctd3) | Lorentz Train Curvature    |
| SCars    | False      | 55.950% | 76.620% | 36.010% | [teuxctd3](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/teuxctd3) | Lorentz Train Curvature    |
| Aircraft | True       | 52.600% | 64.510% | 40.660% | [wtin1ebe](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/wtin1ebe) | Poincare HCD Setup         |
| Aircraft | False      | 50.800% | 59.050% | 42.520% | [wtin1ebe](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/wtin1ebe) | Poincare HCD Setup         |
| CUB      | True       | 00.520% | 01.040% | 00.000% | [hlvo7yqh](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/hlvo7yqh) | Poincare HCD Setup         |
| CUB      | False      | 56.750% | 60.190% | 53.330% | [hlvo7yqh](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/hlvo7yqh) | Poincare HCD Setup         |
| SCars    | True       | 46.210% | 68.770% | 24.460% | [0n9ze8xv](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/0n9ze8xv) | Poincare HCD Setup         |
| SCars    | False      | 41.140% | 60.870% | 28.000% | [0n9ze8xv](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/0n9ze8xv) | Poincare HCD Setup         |
| Aircraft | True       | 41.250% | 54.500% | 27.990% | [3o9glfh8](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/3o9glfh8) | Poincare HCD Angle Only    |
| Aircraft | False      | 43.920% | 55.040% | 32.790% | [3o9glfh8](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/3o9glfh8) | Poincare HCD Angle Only    |
| CUB      | True       | 52.140% | 69.140% | 35.290% | [5qnne9bv](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/5qnne9bv) | Poincare HCD Angle Only    |
| CUB      | False      | 52.520% | 67.200% | 37.970% | [5qnne9bv](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/5qnne9bv) | Poincare HCD Angle Only    |
| SCars    | True       | 40.070% | 64.200% | 17.760% | [yj6ogaf1](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/yj6ogaf1) | Poincare HCD Angle Only    |
| SCars    | False      | 41.190% | 58.360% | 24.630% | [yj6ogaf1](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/yj6ogaf1) | Poincare HCD Angle Only    |



# DINOv2

| Dataset  | Hyp_Kmeans | Acc_All | Acc_Old | Acc_New | wandb                                                                  | Experiment                 |
| -------- | ---------- | ------- | ------- | ------- | ---------------------------------------------------------------------- | -------------------------- |
| Aircraft | False      | 45.480% | 58.570% | 32.370% | [qxzx0a2l](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/qxzx0a2l) | Euclidean 256              |
| CUB      | False      | 55.200% | 69.660% | 40.860% | [f7fd8o9c](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/f7fd8o9c) | Euclidean 256              |
| SCars    | False      | 65.200% | 82.220% | 48.790% | [zzouanev](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/zzouanev) | Euclidean 256              |
| Aircraft | False      | 56.830% | 65.650% | 47.990% | [s22hu696](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/s22hu696) | Euclidean 65536            |
| CUB      | False      | 64.140% | 76.630% | 51.750% | [9pt5bo4v](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9pt5bo4v) | Euclidean 65536            |
| SCars    | False      | 67.120% | 84.120% | 50.720% | [bq6k0tjb](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/bq6k0tjb) | Euclidean 65536            |
| CIFAR10  | False      | 88.970% | 99.160% | 78.780% | [hjl85955](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/hjl85955) | Euclidean 65536            |
| CIFAR100 | False      | 81.700% | 88.310% | 55.250% | [2mshrgdp](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/2mshrgdp) | Euclidean 65536            |
| Aircraft | True       | 61.690% | 66.130% | 57.250% | [jgsafl73](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jgsafl73) | Lorentz HCD Setup          |
| Aircraft | False      | 60.970% | 70.320% | 51.590% | [jgsafl73](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jgsafl73) | Lorentz HCD Setup          |
| CUB      | True       | 73.040% | 83.700% | 62.470% | [9ild8vw2](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9ild8vw2) | Lorentz HCD Setup          |
| CUB      | False      | 72.950% | 82.070% | 63.920% | [9ild8vw2](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9ild8vw2) | Lorentz HCD Setup          |
| SCars    | True       | 76.180% | 90.980% | 61.910% | [y4ewbdoi](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/y4ewbdoi) | Lorentz HCD Setup          |
| SCars    | False      | 72.500% | 86.850% | 58.660% | [y4ewbdoi](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/y4ewbdoi) | Lorentz HCD Setup          |
| CIFAR10  | True       | 72.160% | 98.260% | 46.060% | [4tcfsmny](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/4tcfsmny) | Lorentz HCD Setup          |
| CIFAR10  | False      | 88.040% | 98.040% | 78.040% | [4tcfsmny](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/4tcfsmny) | Lorentz HCD Setup          |
| CIFAR100 | True       | 88.620% | 91.450% | 77.300% | [xud9nwzu](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/xud9nwzu) | Lorentz HCD Setup          |
| CIFAR100 | False      | 85.500% | 91.990% | 62.750% | [xud9nwzu](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/xud9nwzu) | Lorentz HCD Setup          |
| Aircraft | True       | 33.930% | 36.330% | 31.530% | [9pgq944x](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9pgq944x) | Lorentz Frozen Curvature   |
| Aircraft | False      | 66.220% | 76.740% | 55.680% | [9pgq944x](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9pgq944x) | Lorentz Frozen Curvature   |
| CUB      | True       | 68.740% | 84.950% | 52.680% | [r1re2v1l](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/r1re2v1l) | Lorentz Frozen Curvature   |
| CUB      | False      | 71.760% | 79.790% | 63.810% | [r1re2v1l](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/r1re2v1l) | Lorentz Frozen Curvature   |
| SCars    | True       | 75.120% | 89.940% | 60.810% | [0wbw85qv](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/0wbw85qv) | Lorentz Frozen Curvature   |
| SCars    | False      | 72.250% | 88.040% | 57.020% | [0wbw85qv](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/0wbw85qv) | Lorentz Frozen Curvature   |
| Aircraft | True       | 52.960% | 68.170% | 37.720% | [9gy0ghyt](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9gy0ghyt) | Lorentz Train Curvature    |
| Aircraft | False      | 65.500% | 72.120% | 58.860% | [9gy0ghyt](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9gy0ghyt) | Lorentz Train Curvature    |
| CUB      | True       | 49.030% | 66.500% | 31.720% | [arlynhgz](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/arlynhgz) | Lorentz Train Curvature    |
| CUB      | False      | 68.830% | 82.180% | 55.600% | [arlynhgz](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/arlynhgz) | Lorentz Train Curvature    |
| SCars    | True       | 70.660% | 86.650% | 55.240% | [e0ae67f1](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/e0ae67f1) | Lorentz Train Curvature    |
| SCars    | False      | 72.140% | 88.100% | 56.760% | [e0ae67f1](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/e0ae67f1) | Lorentz Train Curvature    |
| Aircraft | True       | 59.110% | 74.460% | 43.720% | [4av8f0gk](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/4av8f0gk) | Poincare HCD Setup         |
| Aircraft | False      | 64.360% | 73.320% | 55.380% | [4av8f0gk](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/4av8f0gk) | Poincare HCD Setup         |
| CUB      | True       | 00.520% | 01.040% | 00.000% | [ldcth3sy](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/ldcth3sy) | Poincare HCD Setup         |
| CUB      | False      | 70.470% | 78.920% | 62.100% | [ldcth3sy](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/ldcth3sy) | Poincare HCD Setup         |
| SCars    | True       | 00.850% | 00.000% | 01.660% | [zecgvqxb](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/zecgvqxb) | Poincare HCD Setup         |
| SCars    | False      | 72.620% | 88.450% | 57.340% | [zecgvqxb](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/zecgvqxb) | Poincare HCD Setup         |
| CIFAR10  | True       | 10.000% | 20.000% | 00.000% | [lhbwps0p](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/lhbwps0p) | Poincare HCD Setup         |
| CIFAR10  | False      | 88.170% | 98.000% | 78.460% | [lhbwps0p](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/lhbwps0p) | Poincare HCD Setup         |
| CIFAR100 | True       | 01.000% | 00.000% | 05.000% | [zmzhwznh](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/zmzhwznh) | Poincare HCD Setup         |
| CIFAR100 | False      | 83.230% | 83.840% | 80.800% | [zmzhwznh](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/zmzhwznh) | Poincare HCD Setup         |
| Aircraft | True       | 00.000% | 00.000% | 00.000% | [k4d48lkz](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/k4d48lkz) | Poincare HCD Angle Only    |
| Aircraft | False      | 00.000% | 00.000% | 00.000% | [k4d48lkz](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/k4d48lkz) | Poincare HCD Angle Only    |
| CUB      | True       | 00.000% | 00.000% | 00.000% | [ziq86ya2](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/ziq86ya2) | Poincare HCD Angle Only    |
| CUB      | False      | 00.000% | 00.000% | 00.000% | [ziq86ya2](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/ziq86ya2) | Poincare HCD Angle Only    |
| SCars    | True       | 00.000% | 00.000% | 00.000% | [x4gqkatf](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/x4gqkatf) | Poincare HCD Angle Only    |
| SCars    | False      | 00.000% | 00.000% | 00.000% | [x4gqkatf](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/x4gqkatf) | Poincare HCD Angle Only    |



Plan:
- Use following models for main results:
  - Euclidean with DINOv2
  - Lorentz with HCD setup and DINOv2
  - Poincare with HCD setup and DINOv2
- Check results of Euclidean embeddings without removing head? This will require extra implementaion, so I am not sure.
- Ablate the need for angle loss based on my hypothesis with Euclidean embeddings making all points lay on a hyper-sphere
- Ablate the different embedding sizes on winning Lorentz model
- Ablate need for decreasing angle loss weight (Low priority)
- Show results with DINO, but that might not be needed, as it barely adds to the discussion. DINOv2 is obviously better and it obviously gets better results.



Ablations:

Only Lorentz, as Poincaré K-Means is very unstable and initialization dependant, and the other paper has already researched Poincaré.

Euclidean clipping:

| Dataset  | Hyp_Kmeans | Acc_All | Acc_Old | Acc_New | wandb                                                                  | Experiment                    |
| -------- | ---------- | ------- | ------- | ------- | ---------------------------------------------------------------------- | ----------------------------- |
| Aircraft | True       | 61.690% | 66.130% | 57.250% | [jgsafl73](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jgsafl73) | Lorentz HCD Setup r=2.3       |
| Aircraft | False      | 60.970% | 70.320% | 51.590% | [jgsafl73](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jgsafl73) | Lorentz HCD Setup r=2.3       |
| CUB      | True       | 73.040% | 83.700% | 62.470% | [9ild8vw2](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9ild8vw2) | Lorentz HCD Setup r=2.3       |
| CUB      | False      | 72.950% | 82.070% | 63.920% | [9ild8vw2](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9ild8vw2) | Lorentz HCD Setup r=2.3       |
| SCars    | True       | 76.180% | 90.980% | 61.910% | [y4ewbdoi](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/y4ewbdoi) | Lorentz HCD Setup r=2.3       |
| SCars    | False      | 72.500% | 86.850% | 58.660% | [y4ewbdoi](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/y4ewbdoi) | Lorentz HCD Setup r=2.3       |
| Aircraft | True       | 65.920% | 80.100% | 51.710% | [jgsafl73](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jgsafl73) | Lorentz HCD Setup No Clipping |
| Aircraft | False      | 65.410% | 72.300% | 58.500% | [jgsafl73](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jgsafl73) | Lorentz HCD Setup No Clipping |
| CUB      | True       | 73.850% | 82.730% | 65.050% | [9ild8vw2](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9ild8vw2) | Lorentz HCD Setup No Clipping |
| CUB      | False      | 71.490% | 80.860% | 62.200% | [9ild8vw2](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9ild8vw2) | Lorentz HCD Setup No Clipping |
| SCars    | True       | 74.930% | 89.310% | 61.060% | [y4ewbdoi](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/y4ewbdoi) | Lorentz HCD Setup No Clipping |
| SCars    | False      | 70.870% | 86.750% | 55.560% | [y4ewbdoi](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/y4ewbdoi) | Lorentz HCD Setup No Clipping |

Angle loss:

| Dataset  | Hyp_Kmeans | Acc_All | Acc_Old | Acc_New | wandb                                                                  | Experiment                       |
| -------- | ---------- | ------- | ------- | ------- | ---------------------------------------------------------------------- | -------------------------------- |
| Aircraft | True       | 61.690% | 66.130% | 57.250% | [jgsafl73](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jgsafl73) | Lorentz HCD Setup                |
| Aircraft | False      | 60.970% | 70.320% | 51.590% | [jgsafl73](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jgsafl73) | Lorentz HCD Setup                |
| CUB      | True       | 73.040% | 83.700% | 62.470% | [9ild8vw2](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9ild8vw2) | Lorentz HCD Setup                |
| CUB      | False      | 72.950% | 82.070% | 63.920% | [9ild8vw2](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9ild8vw2) | Lorentz HCD Setup                |
| SCars    | True       | 76.180% | 90.980% | 61.910% | [y4ewbdoi](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/y4ewbdoi) | Lorentz HCD Setup                |
| SCars    | False      | 72.500% | 86.850% | 58.660% | [y4ewbdoi](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/y4ewbdoi) | Lorentz HCD Setup                |
| Aircraft | True       | 44.790% | 58.270% | 31.290% | [jgsafl73](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jgsafl73) | Lorentz HCD Setup Only Angle     |
| Aircraft | False      | 53.200% | 65.170% | 41.200% | [jgsafl73](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jgsafl73) | Lorentz HCD Setup Only Angle     |
| CUB      | True       | 54.370% | 73.540% | 35.360% | [9ild8vw2](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9ild8vw2) | Lorentz HCD Setup Only Angle     |
| CUB      | False      | 65.430% | 78.880% | 52.100% | [9ild8vw2](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9ild8vw2) | Lorentz HCD Setup Only Angle     |
| SCars    | True       | 56.240% | 76.470% | 36.720% | [y4ewbdoi](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/y4ewbdoi) | Lorentz HCD Setup Only Angle     |
| SCars    | False      | 65.920% | 81.310% | 51.090% | [y4ewbdoi](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/y4ewbdoi) | Lorentz HCD Setup Only Angle     |
| Aircraft | True       | 58.390% | 70.260% | 46.490% | [jgsafl73](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jgsafl73) | Lorentz HCD Setup No Alpha Decay |
| Aircraft | False      | 61.540% | 71.700% | 51.350% | [jgsafl73](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jgsafl73) | Lorentz HCD Setup No Alpha Decay |
| CUB      | True       | 69.380% | 83.950% | 54.950% | [9ild8vw2](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9ild8vw2) | Lorentz HCD Setup No Alpha Decay |
| CUB      | False      | 70.500% | 82.660% | 58.450% | [9ild8vw2](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9ild8vw2) | Lorentz HCD Setup No Alpha Decay |
| SCars    | True       | 70.260% | 86.300% | 54.800% | [y4ewbdoi](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/y4ewbdoi) | Lorentz HCD Setup No Alpha Decay |
| SCars    | False      | 67.850% | 85.310% | 51.010% | [y4ewbdoi](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/y4ewbdoi) | Lorentz HCD Setup No Alpha Decay |

Embedding Dimension:

| Dataset  | Hyp_Kmeans | Acc_All | Acc_Old | Acc_New | wandb                                                                  | Experiment              |
| -------- | ---------- | ------- | ------- | ------- | ---------------------------------------------------------------------- | ----------------------- |
| Aircraft | True       | 57.010% | 67.270% | 46.730% | [27ho66n4](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/27ho66n4) | Lorentz HCD Setup 32    |
| Aircraft | False      | 57.730% | 67.810% | 41.620% | [27ho66n4](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/27ho66n4) | Lorentz HCD Setup 32    |
| CUB      | True       | 54.730% | 81.350% | 50.580% | [c3sx9h77](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/c3sx9h77) | Lorentz HCD Setup 32    |
| CUB      | False      | 68.400% | 78.990% | 57.900% | [c3sx9h77](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/c3sx9h77) | Lorentz HCD Setup 32    |
| SCars    | True       | 70.730% | 85.920% | 56.070% | [8iur5kz7](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/8iur5kz7) | Lorentz HCD Setup 32    |
| SCars    | False      | 70.610% | 85.840% | 55.920% | [8iur5kz7](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/8iur5kz7) | Lorentz HCD Setup 32    |
| Aircraft | True       | 54.130% | 70.200% | 56.040% | [2oycvr2i](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/2oycvr2i) | Lorentz HCD Setup 64    |
| Aircraft | False      | 64.900% | 73.560% | 56.220% | [2oycvr2i](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/2oycvr2i) | Lorentz HCD Setup 64    |
| CUB      | True       | 73.210% | 84.710% | 61.820% | [ov851wgy](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/ov851wgy) | Lorentz HCD Setup 64    |
| CUB      | False      | 72.520% | 81.690% | 63.440% | [ov851wgy](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/ov851wgy) | Lorentz HCD Setup 64    |
| SCars    | True       | 74.580% | 87.540% | 62.080% | [ws1vf8gc](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/ws1vf8gc) | Lorentz HCD Setup 64    |
| SCars    | False      | 72.040% | 83.460% | 61.030% | [ws1vf8gc](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/ws1vf8gc) | Lorentz HCD Setup 64    |
| Aircraft | True       | 65.560% | 72.120% | 58.980% | [jgsafl73](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jgsafl73) | Lorentz HCD Setup 128   |
| Aircraft | False      | 63.910% | 72.000% | 55.800% | [jgsafl73](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jgsafl73) | Lorentz HCD Setup 128   |
| CUB      | True       | 73.010% | 84.120% | 61.990% | [9ild8vw2](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9ild8vw2) | Lorentz HCD Setup 128   |
| CUB      | False      | 71.140% | 82.140% | 60.240% | [9ild8vw2](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9ild8vw2) | Lorentz HCD Setup 128   |
| SCars    | True       | 74.570% | 89.690% | 59.980% | [y4ewbdoi](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/y4ewbdoi) | Lorentz HCD Setup 128   |
| SCars    | False      | 72.090% | 88.220% | 56.540% | [y4ewbdoi](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/y4ewbdoi) | Lorentz HCD Setup 128   |
| Aircraft | True       | 61.690% | 66.130% | 57.250% | [jgsafl73](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jgsafl73) | Lorentz HCD Setup 256   |
| Aircraft | False      | 60.970% | 70.320% | 51.590% | [jgsafl73](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jgsafl73) | Lorentz HCD Setup 256   |
| CUB      | True       | 73.040% | 83.700% | 62.470% | [9ild8vw2](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9ild8vw2) | Lorentz HCD Setup 256   |
| CUB      | False      | 72.950% | 82.070% | 63.920% | [9ild8vw2](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9ild8vw2) | Lorentz HCD Setup 256   |
| SCars    | True       | 76.180% | 90.980% | 61.910% | [y4ewbdoi](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/y4ewbdoi) | Lorentz HCD Setup 256   |
| SCars    | False      | 72.500% | 86.850% | 58.660% | [y4ewbdoi](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/y4ewbdoi) | Lorentz HCD Setup 256   |
| Aircraft | True       | 58.420% | 61.870% | 54.950% | [jgsafl73](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jgsafl73) | Lorentz HCD Setup 768   |
| Aircraft | False      | 61.990% | 70.380% | 53.570% | [jgsafl73](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jgsafl73) | Lorentz HCD Setup 768   |
| CUB      | True       | 70.610% | 87.520% | 55.680% | [9ild8vw2](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9ild8vw2) | Lorentz HCD Setup 768   |
| CUB      | False      | 70.070% | 79.370% | 60.860% | [9ild8vw2](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9ild8vw2) | Lorentz HCD Setup 768   |
| SCars    | True       | 75.950% | 90.350% | 62.060% | [y4ewbdoi](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/y4ewbdoi) | Lorentz HCD Setup 768   |
| SCars    | False      | 73.770% | 85.280% | 62.670% | [y4ewbdoi](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/y4ewbdoi) | Lorentz HCD Setup 768   |
| Aircraft | True       | 59.320% | 77.400% | 50.150% | [jgsafl73](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jgsafl73) | Lorentz HCD Setup 65536 |
| Aircraft | False      | 59.710% | 65.470% | 53.930% | [jgsafl73](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jgsafl73) | Lorentz HCD Setup 65536 |
| CUB      | True       | 72.560% | 84.360% | 60.860% | [9ild8vw2](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9ild8vw2) | Lorentz HCD Setup 65536 |
| CUB      | False      | 72.210% | 81.450% | 63.060% | [9ild8vw2](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9ild8vw2) | Lorentz HCD Setup 65536 |
| SCars    | True       | 75.540% | 89.060% | 62.500% | [y4ewbdoi](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/y4ewbdoi) | Lorentz HCD Setup 65536 |
| SCars    | False      | 74.570% | 88.600% | 61.030% | [y4ewbdoi](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/y4ewbdoi) | Lorentz HCD Setup 65536 |

Learning Curvature

| Dataset  | Hyp_Kmeans | Acc_All | Acc_Old | Acc_New | wandb                                                                  | Experiment                         |
| -------- | ---------- | ------- | ------- | ------- | ---------------------------------------------------------------------- | ---------------------------------- |
| Aircraft | True       | 61.690% | 66.130% | 57.250% | [jgsafl73](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jgsafl73) | Lorentz HCD Setup                  |
| Aircraft | False      | 60.970% | 70.320% | 51.590% | [jgsafl73](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jgsafl73) | Lorentz HCD Setup                  |
| CUB      | True       | 73.040% | 83.700% | 62.470% | [9ild8vw2](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9ild8vw2) | Lorentz HCD Setup                  |
| CUB      | False      | 72.950% | 82.070% | 63.920% | [9ild8vw2](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9ild8vw2) | Lorentz HCD Setup                  |
| SCars    | True       | 76.180% | 90.980% | 61.910% | [y4ewbdoi](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/y4ewbdoi) | Lorentz HCD Setup                  |
| SCars    | False      | 72.500% | 86.850% | 58.660% | [y4ewbdoi](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/y4ewbdoi) | Lorentz HCD Setup                  |
| Aircraft | True       | 61.360% | 75.840% | 46.850% | [jgsafl73](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jgsafl73) | Lorentz HCD Setup Learn Curv       |
| Aircraft | False      | 63.370% | 74.820% | 51.890% | [jgsafl73](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jgsafl73) | Lorentz HCD Setup Learn Curv       |
| CUB      | True       | 64.070% | 81.900% | 46.390% | [9ild8vw2](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9ild8vw2) | Lorentz HCD Setup Learn Curv       |
| CUB      | False      | 67.220% | 81.100% | 53.470% | [9ild8vw2](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9ild8vw2) | Lorentz HCD Setup Learn Curv       |
| SCars    | True       | 66.930% | 83.690% | 50.770% | [y4ewbdoi](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/y4ewbdoi) | Lorentz HCD Setup Learn Curv       |
| SCars    | False      | 67.070% | 81.480% | 53.160% | [y4ewbdoi](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/y4ewbdoi) | Lorentz HCD Setup Learn Curv       |
| Aircraft | True       | 58.600% | 66.730% | 50.450% | [83tmrkj9](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/83tmrkj9) | Lorentz HCD Setup Learn Curv Fixed |
| Aircraft | False      | 63.070% | 75.840% | 50.270% | [83tmrkj9](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/83tmrkj9) | Lorentz HCD Setup Learn Curv Fixed |
| CUB      | True       | 61.240% | 80.030% | 42.610% | [leutd2es](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/leutd2es) | Lorentz HCD Setup Learn Curv Fixed |
| CUB      | False      | 64.700% | 78.120% | 51.410% | [leutd2es](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/leutd2es) | Lorentz HCD Setup Learn Curv Fixed |
| SCars    | True       | 66.840% | 84.320% | 49.990% | [7wls8ewz](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/7wls8ewz) | Lorentz HCD Setup Learn Curv Fixed |
| SCars    | False      | 68.160% | 84.570% | 52.330% | [7wls8ewz](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/7wls8ewz) | Lorentz HCD Setup Learn Curv Fixed |

AdamW

| Dataset  | Hyp_Kmeans | Acc_All | Acc_Old | Acc_New | wandb                                                                  | Experiment             |
| -------- | ---------- | ------- | ------- | ------- | ---------------------------------------------------------------------- | ---------------------- |
| Aircraft | True       | 61.690% | 66.130% | 57.250% | [jgsafl73](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jgsafl73) | Lorentz HCD Setup      |
| Aircraft | False      | 60.970% | 70.320% | 51.590% | [jgsafl73](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jgsafl73) | Lorentz HCD Setup      |
| CUB      | True       | 73.040% | 83.700% | 62.470% | [9ild8vw2](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9ild8vw2) | Lorentz HCD Setup      |
| CUB      | False      | 72.950% | 82.070% | 63.920% | [9ild8vw2](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9ild8vw2) | Lorentz HCD Setup      |
| SCars    | True       | 76.180% | 90.980% | 61.910% | [y4ewbdoi](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/y4ewbdoi) | Lorentz HCD Setup      |
| SCars    | False      | 72.500% | 86.850% | 58.660% | [y4ewbdoi](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/y4ewbdoi) | Lorentz HCD Setup      |
| Aircraft | True       | 41.370% | 60.850% | 21.860% | [jgsafl73](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jgsafl73) | Lorentz HCD Setup Adam |
| Aircraft | False      | 42.090% | 48.980% | 35.200% | [jgsafl73](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jgsafl73) | Lorentz HCD Setup Adam |
| CUB      | True       | 29.220% | 39.940% | 18.590% | [9ild8vw2](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9ild8vw2) | Lorentz HCD Setup Adam |
| CUB      | False      | 30.690% | 40.190% | 21.270% | [9ild8vw2](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9ild8vw2) | Lorentz HCD Setup Adam |
| SCars    | True       | 36.140% | 55.290% | 17.660% | [y4ewbdoi](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/y4ewbdoi) | Lorentz HCD Setup Adam |
| SCars    | False      | 34.290% | 46.630% | 22.380% | [y4ewbdoi](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/y4ewbdoi) | Lorentz HCD Setup Adam |