# Retreat Demo Cartpole

Create Conda environment

```bash
conda create -yn cartpole python=3.8
conda activate cartpole
```


```bash
sudo apt install build-essential
pip install -r requirements.txt
```

```bash
cd ./gym_env/
pip install -e .
```

## Running pretrained model

```bash
python ./scripts/gym_test.py
```



## Compiling Spike Device



TODO:

- move current baremetal C code out of chipyard/tests

- debug why model is not working in spike, might be weight mismatch etc. need to give fixed input and cross check with np version



