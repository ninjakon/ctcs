# Model Test Runner
### CTCS 2024/25 - UIBK

## Running the test runner

It is recommended to use a Python virtual environment.
The following should set it up, activate it and install the requirements.

```
python3 -m venv .venv
source ./.venv/bin/activate
pip3 install -r requirements.txt # this may take some time
```

After setting up all of that you can run it by simply calling `main.py`.
If you don't have Cuda you can also use Torch but the models should be able to run on CPU which you can do by simply not passing any argument for `--device`.

```
python3 main.py --device cuda
```
OR
```
python3 main.py --device torch
```
OR
```
python3 main.py --device cpu
```

## Adding new models

// TODO
