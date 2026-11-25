# Install

ServeCast needs Python 3.11 or newer. To predict, it runs on Linux and macOS with no GPU. To measure on a GPU
(`servecast profile`), it needs Linux and an NVIDIA GPU of compute capability 8.0 or newer listed by
`servecast list gpus`, with driver 580 or newer.

## From PyPI

```bash
# to predict
pip install servecast
# to predict, and to measure on an NVIDIA GPU under Linux
# (adds vLLM 0.25.1 and NVIDIA's NVML library)
pip install 'servecast[profile]'
```

## From a clone of the repository

```bash
git clone https://github.com/servecast/servecast.git
cd servecast
# to predict
pip install .
# to predict, and to measure on an NVIDIA GPU under Linux
pip install '.[profile]'
```

## Check

```bash
servecast doctor
```

`servecast doctor` names the GPU and the versions of vLLM and PyTorch, and says whether the profile commands can run on
this computer. When they cannot, it says why, for example a GPU older than compute capability 8.0 or a driver too old
for the PyTorch that vLLM installs.
