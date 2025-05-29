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
| Aircraft | False      | 45.480% | 58.570% | 32.370% | [qxzx0a2l](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/qxzx0a2l) | Euclidean                  |
| CUB      | False      | 55.200% | 69.660% | 40.860% | [f7fd8o9c](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/f7fd8o9c) | Euclidean                  |
| SCars    | False      | 65.200% | 82.220% | 48.790% | [zzouanev](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/zzouanev) | Euclidean                  |
| Aircraft | True       | 61.690% | 66.130% | 57.250% | [jgsafl73](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jgsafl73) | Lorentz HCD Setup          |
| Aircraft | False      | 60.970% | 70.320% | 51.590% | [jgsafl73](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jgsafl73) | Lorentz HCD Setup          |
| CUB      | True       | 73.040% | 83.700% | 62.470% | [9ild8vw2](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9ild8vw2) | Lorentz HCD Setup          |
| CUB      | False      | 72.950% | 82.070% | 63.920% | [9ild8vw2](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/9ild8vw2) | Lorentz HCD Setup          |
| SCars    | True       | 76.180% | 90.980% | 61.910% | [y4ewbdoi](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/y4ewbdoi) | Lorentz HCD Setup          |
| SCars    | False      | 72.500% | 86.850% | 58.660% | [y4ewbdoi](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/y4ewbdoi) | Lorentz HCD Setup          |
| Aircraft | True       | 33.930% | 36.330% | 31.530% | [ob4xrtq1](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/ob4xrtq1) | Lorentz Frozen Curvature   |
| Aircraft | False      | 66.220% | 76.740% | 55.680% | [ob4xrtq1](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/ob4xrtq1) | Lorentz Frozen Curvature   |
| CUB      | True       | 68.740% | 84.950% | 52.680% | [3sjepj1t](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/3sjepj1t) | Lorentz Frozen Curvature   |
| CUB      | False      | 71.760% | 79.790% | 63.810% | [3sjepj1t](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/3sjepj1t) | Lorentz Frozen Curvature   |
| SCars    | True       | 75.120% | 89.940% | 60.810% | [teuxctd3](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/teuxctd3) | Lorentz Frozen Curvature   |
| SCars    | False      | 72.250% | 88.040% | 57.020% | [teuxctd3](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/teuxctd3) | Lorentz Frozen Curvature   |
| Aircraft | True       | 52.960% | 68.170% | 37.720% | [ob4xrtq1](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/ob4xrtq1) | Lorentz Train Curvature    |
| Aircraft | False      | 65.500% | 72.120% | 58.860% | [ob4xrtq1](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/ob4xrtq1) | Lorentz Train Curvature    |
| CUB      | True       | 49.030% | 66.500% | 31.720% | [3sjepj1t](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/3sjepj1t) | Lorentz Train Curvature    |
| CUB      | False      | 68.830% | 82.180% | 55.600% | [3sjepj1t](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/3sjepj1t) | Lorentz Train Curvature    |
| SCars    | True       | 70.660% | 86.650% | 55.240% | [teuxctd3](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/teuxctd3) | Lorentz Train Curvature    |
| SCars    | False      | 72.140% | 88.100% | 56.760% | [teuxctd3](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/teuxctd3) | Lorentz Train Curvature    |
| Aircraft | True       | 59.110% | 74.460% | 43.720% | [wtin1ebe](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/wtin1ebe) | Poincare HCD Setup         |
| Aircraft | False      | 64.360% | 73.320% | 55.380% | [wtin1ebe](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/wtin1ebe) | Poincare HCD Setup         |
| CUB      | True       | 00.520% | 01.040% | 00.000% | [hlvo7yqh](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/hlvo7yqh) | Poincare HCD Setup         |
| CUB      | False      | 70.470% | 78.920% | 62.100% | [hlvo7yqh](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/hlvo7yqh) | Poincare HCD Setup         |
| SCars    | True       | 00.850% | 00.000% | 01.660% | [0n9ze8xv](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/0n9ze8xv) | Poincare HCD Setup         |
| SCars    | False      | 72.620% | 88.450% | 57.340% | [0n9ze8xv](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/0n9ze8xv) | Poincare HCD Setup         |
| Aircraft | True       | 00.000% | 00.000% | 00.000% | [3o9glfh8](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/3o9glfh8) | Poincare HCD Angle Only    |
| Aircraft | False      | 00.000% | 00.000% | 00.000% | [3o9glfh8](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/3o9glfh8) | Poincare HCD Angle Only    |
| CUB      | True       | 00.000% | 00.000% | 00.000% | [5qnne9bv](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/5qnne9bv) | Poincare HCD Angle Only    |
| CUB      | False      | 00.000% | 00.000% | 00.000% | [5qnne9bv](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/5qnne9bv) | Poincare HCD Angle Only    |
| SCars    | True       | 00.000% | 00.000% | 00.000% | [yj6ogaf1](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/yj6ogaf1) | Poincare HCD Angle Only    |
| SCars    | False      | 00.000% | 00.000% | 00.000% | [yj6ogaf1](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/yj6ogaf1) | Poincare HCD Angle Only    |



Plan:
- To be decided