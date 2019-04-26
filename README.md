# y2 Worker Client

The worker service for the y2 cluster.

## Setup

- `pip3 install -r requirements.txt` to install the website's required packages

## Virtual Environments

It is highly recommended to work using virtual environments in Python.

- Install virtualenvwrapper: `pip install virtualenvwrapper`.
- Create a new virtual environment for the y2 worker: `mkvirtualenv y2`.
- Activate the y2 virtual environment: `workon y2`
- Deactivate the virtual environment: `deactivate`

## Package Management

Please only issue those commands inside a virtual environment.

- `pip3 install -r requirements.txt` to install all required dependencies
- `pip3 install <package_name>` to install a new package (you should probably add the newly install package to the project's list of dependencies)
- `pip3 freeze --local -r requirements.txt > requirements.txt` to update the list of requirements (without deleting any entry already present in the `requirements.txt` file)

## Compiling the CUDA Code

To compile the CUDA code, run `python3 setup.py install` inside the cuda folder.
You will need both a C++ compiler installed and a working CUDA installation (`nvcc`).