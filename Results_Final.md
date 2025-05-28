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
| Aircraft | False      |         |         |         | [8fpqckq6](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/8fpqckq6) | Euclidean                  |
| CUB      | False      |         |         |         | [cb93l49j](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/cb93l49j) | Euclidean                  |
| SCars    | False      |         |         |         | [ce4ultl0](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/ce4ultl0) | Euclidean                  |
| Aircraft | True       | 28.230% | 30.190% | 27.620% | [gkx5u4iz](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/gkx5u4iz) | Lorentz HCD Setup          |
| Aircraft | False      | 36.610% | 38.240% | 35.800% | [gkx5u4iz](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/gkx5u4iz) | Lorentz HCD Setup          |
| CUB      | True       | 39.750% | 40.690% | 39.270% | [979w86nu](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/979w86nu) | Lorentz HCD Setup          |
| CUB      | False      | 53.960% | 53.640% | 54.120% | [979w86nu](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/979w86nu) | Lorentz HCD Setup          |
| SCars    | True       | 30.260% | 49.580% | 20.930% | [jlve0elr](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jlve0elr) | Lorentz HCD Setup          |
| SCars    | False      | 39.450% | 59.970% | 29.540% | [jlve0elr](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jlve0elr) | Lorentz HCD Setup          |
| Aircraft | True       | 25.370% | 25.450% | 25.340% | [ob4xrtq1](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/ob4xrtq1) | Lorentz Frozen Curvature   |
| Aircraft | False      | 51.050% | 54.800% | 49.180% | [ob4xrtq1](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/ob4xrtq1) | Lorentz Frozen Curvature   |
| CUB      | True       | 29.380% | 42.490% | 22.820% | [3sjepj1t](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/3sjepj1t) | Lorentz Frozen Curvature   |
| CUB      | False      | 55.360% | 59.910% | 53.090% | [3sjepj1t](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/3sjepj1t) | Lorentz Frozen Curvature   |
| SCars    | True       | 26.920% | 50.620% | 15.470% | [teuxctd3](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/teuxctd3) | Lorentz Frozen Curvature   |
| SCars    | False      | 45.410% | 64.820% | 36.040% | [teuxctd3](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/teuxctd3) | Lorentz Frozen Curvature   |
| Aircraft | True       | 26.710% | 24.550% | 27.800% | [ob4xrtq1](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/ob4xrtq1) | Lorentz Train Curvature    |
| Aircraft | False      | 46.210% | 53.720% | 42.460% | [ob4xrtq1](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/ob4xrtq1) | Lorentz Train Curvature    |
| CUB      | True       | 29.560% | 40.090% | 24.290% | [3sjepj1t](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/3sjepj1t) | Lorentz Train Curvature    |
| CUB      | False      | 53.960% | 58.640% | 51.620% | [3sjepj1t](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/3sjepj1t) | Lorentz Train Curvature    |
| SCars    | True       | 29.980% | 49.650% | 20.540% | [teuxctd3](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/teuxctd3) | Lorentz Train Curvature    |
| SCars    | False      | 52.180% | 71.410% | 42.890% | [teuxctd3](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/teuxctd3) | Lorentz Train Curvature    |
| Aircraft | True       | 03.120% | 04.020% | 02.670% | [wtin1ebe](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/wtin1ebe) | Poincare HCD Setup         |
| Aircraft | False      | 47.250% | 57.980% | 41.890% | [wtin1ebe](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/wtin1ebe) | Poincare HCD Setup         |
| CUB      | True       | 01.000% | 00.000% | 01.500% | [hlvo7yqh](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/hlvo7yqh) | Poincare HCD Setup         |
| CUB      | False      | 52.000% | 47.630% | 54.190% | [hlvo7yqh](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/hlvo7yqh) | Poincare HCD Setup         |
| SCars    | True       | 01.110% | 00.050% | 01.620% | [0n9ze8xv](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/0n9ze8xv) | Poincare HCD Setup         |
| SCars    | False      | 41.760% | 58.620% | 33.620% | [0n9ze8xv](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/0n9ze8xv) | Poincare HCD Setup         |
| Aircraft | True       | 03.340% | 04.620% | 03.000% | [3o9glfh8](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/3o9glfh8) | Poincare HCD Angle Only    |
| Aircraft | False      | 41.690% | 44.360% | 40.360% | [3o9glfh8](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/3o9glfh8) | Poincare HCD Angle Only    |
| CUB      | True       | 01.780% | 02.270% | 01.530% | [5qnne9bv](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/5qnne9bv) | Poincare HCD Angle Only    |
| CUB      | False      | 50.270% | 51.770% | 49.520% | [5qnne9bv](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/5qnne9bv) | Poincare HCD Angle Only    |
| SCars    | True       | 02.100% | 02.250% | 02.030% | [yj6ogaf1](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/yj6ogaf1) | Poincare HCD Angle Only    |
| SCars    | False      | 37.500% | 55.320% | 28.890% | [yj6ogaf1](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/yj6ogaf1) | Poincare HCD Angle Only    |



# DINOv2

| Dataset  | Hyp_Kmeans | Acc_All | Acc_Old | Acc_New | wandb                                                                  | Experiment                 |
| -------- | ---------- | ------- | ------- | ------- | ---------------------------------------------------------------------- | -------------------------- |
| Aircraft | False      |         |         |         | [8fpqckq6](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/qxzx0a2l) | Euclidean                  |
| CUB      | False      |         |         |         | [cb93l49j](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/f7fd8o9c) | Euclidean                  |
| SCars    | False      |         |         |         | [ce4ultl0](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/zzouanev) | Euclidean                  |
| Aircraft | True       | 28.230% | 30.190% | 27.620% | [gkx5u4iz](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/gkx5u4iz) | Lorentz HCD Setup          |
| Aircraft | False      | 36.610% | 38.240% | 35.800% | [gkx5u4iz](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/gkx5u4iz) | Lorentz HCD Setup          |
| CUB      | True       | 39.750% | 40.690% | 39.270% | [979w86nu](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/979w86nu) | Lorentz HCD Setup          |
| CUB      | False      | 53.960% | 53.640% | 54.120% | [979w86nu](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/979w86nu) | Lorentz HCD Setup          |
| SCars    | True       | 30.260% | 49.580% | 20.930% | [jlve0elr](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jlve0elr) | Lorentz HCD Setup          |
| SCars    | False      | 39.450% | 59.970% | 29.540% | [jlve0elr](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/jlve0elr) | Lorentz HCD Setup          |
| Aircraft | True       | 25.370% | 25.450% | 25.340% | [ob4xrtq1](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/ob4xrtq1) | Lorentz Frozen Curvature   |
| Aircraft | False      | 51.050% | 54.800% | 49.180% | [ob4xrtq1](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/ob4xrtq1) | Lorentz Frozen Curvature   |
| CUB      | True       | 29.380% | 42.490% | 22.820% | [3sjepj1t](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/3sjepj1t) | Lorentz Frozen Curvature   |
| CUB      | False      | 55.360% | 59.910% | 53.090% | [3sjepj1t](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/3sjepj1t) | Lorentz Frozen Curvature   |
| SCars    | True       | 26.920% | 50.620% | 15.470% | [teuxctd3](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/teuxctd3) | Lorentz Frozen Curvature   |
| SCars    | False      | 45.410% | 64.820% | 36.040% | [teuxctd3](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/teuxctd3) | Lorentz Frozen Curvature   |
| Aircraft | True       | 26.710% | 24.550% | 27.800% | [ob4xrtq1](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/ob4xrtq1) | Lorentz Train Curvature    |
| Aircraft | False      | 46.210% | 53.720% | 42.460% | [ob4xrtq1](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/ob4xrtq1) | Lorentz Train Curvature    |
| CUB      | True       | 29.560% | 40.090% | 24.290% | [3sjepj1t](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/3sjepj1t) | Lorentz Train Curvature    |
| CUB      | False      | 53.960% | 58.640% | 51.620% | [3sjepj1t](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/3sjepj1t) | Lorentz Train Curvature    |
| SCars    | True       | 29.980% | 49.650% | 20.540% | [teuxctd3](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/teuxctd3) | Lorentz Train Curvature    |
| SCars    | False      | 52.180% | 71.410% | 42.890% | [teuxctd3](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/teuxctd3) | Lorentz Train Curvature    |
| Aircraft | True       | 03.120% | 04.020% | 02.670% | [wtin1ebe](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/wtin1ebe) | Poincare HCD Setup         |
| Aircraft | False      | 47.250% | 57.980% | 41.890% | [wtin1ebe](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/wtin1ebe) | Poincare HCD Setup         |
| CUB      | True       | 01.000% | 00.000% | 01.500% | [hlvo7yqh](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/hlvo7yqh) | Poincare HCD Setup         |
| CUB      | False      | 52.000% | 47.630% | 54.190% | [hlvo7yqh](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/hlvo7yqh) | Poincare HCD Setup         |
| SCars    | True       | 01.110% | 00.050% | 01.620% | [0n9ze8xv](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/0n9ze8xv) | Poincare HCD Setup         |
| SCars    | False      | 41.760% | 58.620% | 33.620% | [0n9ze8xv](https://wandb.ai/mohamaddalal/Hyperbolic_GCD/runs/0n9ze8xv) | Poincare HCD Setup         |