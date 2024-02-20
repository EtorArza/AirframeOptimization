### Setup

Create a venv, or a conda environment, and activate it. If using environment `airframes`, just follow the instructions there and that will create a conda environment (rlgpu).


#### airframes

1) Follow the installation instructions in
https://github.com/ntnu-arl/aerial_gym_simulator and https://github.com/ntnu-arl/aerial_gym_dev. Requires a NVIDIA gpu and installing `CUDA`, `isaacgym` and all of that fun stuff. 

2) Change the first line in `src/problem_airframes.py` with the path of `aerial_gym_dev` source.




#### general

```
# Install custom supervenn package
pip install -e git+https://github.com/EtorArza/supervenn.git#egg=supervenn
```