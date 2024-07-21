# Robotics SoC Demo: Cartpole

## Setup

```bash
git clone git@github.com:ucb-bar/Robotics-Demo-Cartpole.git
cd ./Robotics-Demo-Cartpole/
```

```bash
conda create -yp ./.conda-env/ python=3.10
```

```bash
conda activate ./.conda-env/
```

```bash
pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit-sdk isaacsim-extscache-kit isaacsim-app --extra-index-url https://pypi.nvidia.com
```

```bash
./isaaclab.sh --install all
```

## Train
```bash
python source/standalone/workflows/rsl_rl/train.py --task Isaac-Cartpole-v0
```

## Visualize



## Exporting the model to Baremetal-IDE

```bash
python source/standalone/workflows/rsl_rl/export.py --task Isaac-Cartpole-v0 --headless
```



