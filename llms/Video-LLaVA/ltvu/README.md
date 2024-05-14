## What is this package `ltvu`?

## Installation

### 1, Video-LLaVA's Requirements
Proceed to the official [README](README.md) of this repo and follow the installation instructions.

To validate your installation, run this.
```bash
python -m videollava.serve.cli --model-path "LanguageBind/Video-LLaVA-7B" --file "path/to/your/video.mp4" --load-4bit
```

### 2. `pyslurm`

Please check the right version corresponding to SLURM installed

```bash
# Check SLURM version. Note that it may differ based on your system.
scontrol --version  # slurm 22.05.2 for me

# download the repo
git clone https://github.com/PySlurm/pyslurm.git && cd pyslurm

# export env vars regarding SLURM, ask your admins
export SLURM_INCLUDE_DIR=/usr/local/include/slurm
export SLURM_LIB_DIR=/usr/local/lib/slurm

# build it, it may take several minutes
scripts/build.sh

# install it under this python env
pip install .
```

### 3. Others

```bash
pip install submitit
```

## Extract Captions

### Short-term

```bash
cd ltvu  # now you are in ./Video-LLaVA/ltvu

# extract ones with submitit
# requires SLURM and this repo to be in a shared storage like a NAS
python -B -m ltvu.experiments.feasibility_check.short_term_saliency
```
