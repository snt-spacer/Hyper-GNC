

![FP](docs/source/_static/hypergnc.png)

---

# Learning Adaptive Multi-Task Guidance, Navigation, and Control via Hypernetworks

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)


Autonomous free-flying robots require versatile control policies that can adapt to varying mission objectives under strict resource constraints. We introduce HYPER-GNC, a framework that utilizes hypernetworks to map semantic embeddings to the weight space of a deep actor-critic policy. This approach enables a single, compact controller to master a diverse suite of Guidance, Navigation, and Control tasks (track velocities, docking, navigation with obstacles, and inspection) without the need for task-specific retraining. We demonstrate that our semantic manifold formulation facilitates zero-shot generalization to novel mission profiles. Extensive experimental results show that HYPER-GNC maintains stability under significant physical perturbations and external body wrenches. Finally, we demonstrate the bridge from simulation to reality by deploying our policy on a physical satellite emulator. To support reproducibility, our code and trained models are made publicly available.


## Getting Started

### Clone the repo and go to the corresponding branch:
```
git clone 
cd Hyper-GNC
```

### Build and start the docker
```
./docker/container.py build
./docker/container.py start
```

### Train

```
./scripts/reinforcement_learning/rsl_rl/control_train_hypernet.sh
```

### Eval
You can download the weights form the google drive and add them inside the `logs` folder of the docker. [Download link](https://drive.google.com/drive/folders/1S7xDNgK4fZporHZsNk_Xj_RW-0p9lY61?usp=sharing).

```
./scripts/reinforcement_learning/rsl_rl/control_eval_hypernet.sh
```
