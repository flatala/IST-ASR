# PHI4 Multimodal on DelftBlue

Full installation:
```sh
module load miniconda3
module load 2023r1
module load cuda/12
module load ffmpeg
conda create -n PHI4-ASR python-3.10
cd phi4
conda activate PHI4-ASR
pip install -r phi4_requirements.txt
```

## Requirements

- Minimum Python 3.10
- Minimum CUDA 12.0
- Tested for `Python 3.10.16`, `Conda 4.5.x`, `CUDA 12.5`
- Python dependencies in `phi4_requirements.txt`. Install in fresh and activated `conda` (py3.10) environment, **using pip**
- `flash_attn` package might fail on the first try and is therefore commented. Because it will download the source and build locally, and it requires some CUDA stuff to work.


To install `flash_attn`, follow:
```sh
module load 2023r1
module load cuda/12
module load ffmpeg
pip install flash_attn==2.7.4.post1
```


## Troubleshooting

### sympy not found:

Reinstall transformers and sympy
```sh
pip uninstall sympy
pip uninstall transformers
pip install sympy==1.13.1
pip install -U transformers
```

