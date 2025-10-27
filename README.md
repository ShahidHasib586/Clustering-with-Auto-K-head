# Clustering with Auto-K (AND)

## AND: Anchor Neighbourhood Discovery

Accepted by 36th International Conference on Machine Learning (ICML 2019).

PyTorch implementation of Unsupervised Deep Learning by Neighbourhood Discovery.

### Highlight

- We propose the idea of exploiting local neighbourhoods for unsupervised deep learning. This strategy preserves the capability of clustering for class boundary inference whilst minimising the negative impact of class inconsistency typically encountered in clusters.
- We formulate an Anchor Neighbourhood Discovery (AND) approach to progressive unsupervised deep learning. The AND model not only generalises the idea of sample specificity learning, but also additionally considers the originally missing sample-to-sample correlation during model learning by a novel neighbourhood supervision design.
- We further introduce a curriculum learning algorithm to gradually perform neighbourhood discovery for maximising the class consistency of neighbourhoods therefore enhancing the unsupervised learning capability.

### Main results

The proposed AND model was evaluated on four object image classification datasets including CIFAR 10/100, SVHN and ImageNet12. Results are shown at the following tables:

## Reproduction

### Requirements

Python 3.10 or newer and PyTorch 2.4 (or any later 2.x release tested against this project) are required. Please refer to `requirements.yaml` for the full Conda environment specification. The Conda environment we used for the experiments can also be rebuilt according to it.

#### Install with Conda (not recommended)

```bash

# set up the environment
# From home dir
cd ~

# install micromamba (single binary, no sudo)
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

# Initialize for bash (use -r, not -p)
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
./bin/micromamba shell init -s bash -r "$MAMBA_ROOT_PREFIX"

# Reload your shell so "micromamba" is on PATH
exec $SHELL -l
# (If it still says command not found: source ~/.bashrc)

# Create & activate a Python 3.10 env
micromamba create -y -n and310 python=3.10
micromamba activate and310

# install torch/vision/audio first (CUDA 11.3 builds)
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 \
  -f https://download.pytorch.org/whl/torch_stable.html



# clone the repository
git clone https://github.com/ShahidHasib586/Clustering-with-Auto-K-head.git
cd Clustering-with-Auto-K-head
# use plain venv
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt

micromamba activate and310

#Torch + TorchVision (GPU CUDA 11.3 for P100). If no NVIDIA driver, use the CPU block below instead.
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 \
  -f https://download.pytorch.org/whl/torch_stable.html

# (If you don’t have a working NVIDIA driver: use CPU wheels)
# pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1+cpu \
#   -f https://download.pytorch.org/whl/torch_stable.html

# Align the NumPy/SciPy stack to pre-NumPy-2 era
pip install --upgrade --no-deps numpy==1.23.5
pip install --upgrade scipy==1.10.1 scikit-learn==1.1.3 scikit-image==0.19.3 matplotlib==3.5.3 Pillow==9.5.0

pip install jinja2 typeguard

# create the environment described in requirements.yaml
#conda env create -f requirements.yaml
#conda activate and-env
```

#### Install with pip (minimal)

```bash
python3 main.py --cfgs configs/base.yaml configs/cifar10.yaml \
  --sess-dir sessions --dataset cifar10 --network ResNet18

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "torch>=2.4.0,<2.6" "torchvision>=0.19.0,<0.21"
pip install numpy scipy scikit-learn scikit-image pillow matplotlib \
    pyyaml prettytable tensorboard tensorboardx faiss-cpu==1.7.4
```

> **CUDA users:** replace the `cpuonly` dependency in `requirements.yaml` with `pytorch-cuda=<cuda_version>` (e.g. `pytorch-cuda=11.8`) and install the matching `faiss-gpu` wheel (`pip install faiss-gpu==1.7.4`).

### Usage

1. Download datasets and store them in `/path/to/AND/data`. (Soft link is recommended to avoid redundant copies of datasets.) For example, if your CIFAR-10 archive already exists at `/datasets/cifar10`, you can link it into the expected location:

   ```bash
   mkdir -p data
   ln -s /datasets/cifar10 data/cifar10
   ```
2. Launch training via the session runner to automatically manage checkpoints and logs:

   ```bash
    python3 main.py --cfgs configs/base.yaml configs/cifar10.yaml \
    --sess-dir sessions --dataset cifar10 --network ResNet18

   ```

   This command reproduces the reported ResNet18 result on CIFAR10. Checkpoints, TensorBoard logs and configuration snapshots are written under `sessions/<timestamp>/`.

3. To evaluate a saved checkpoint without further training, point `--resume` to the checkpoint path and add `--test-only`:

   ```bash
   python3 main.py --cfgs configs/base.yaml configs/cifar10.yaml \
       --resume sessions/<timestamp>/checkpoint/0000-XXXXX.ckpt --test-only
   ```

4. Running on GPUs: the code selects CUDA automatically when available. Use `--gpus` to pick devices explicitly, e.g. `--gpus 0` or `--gpus 0,1`.

5. Hyper-parameter tuning: override any option at the command line. For example, to explore a longer curriculum with a higher learning rate:

   ```bash
   python3 main.py --cfgs configs/base.yaml configs/cifar10.yaml \
       --max-round 7 --max-epoch 240 --base-lr 0.05 --batch-size 256
   ```

   All overrides are logged to `sessions/<timestamp>/config.yaml` for future reference.

6. **Enable the Auto-K head:** turn on automatic cluster-count estimation by adding `--auto-k-enable` to your training command. The default `consensus` strategy combines multiple heuristics and writes its vote to TensorBoard:

   ```bash
   python3 main.py --cfgs configs/base.yaml configs/cifar10.yaml \
       --auto-k-enable --auto-k-method consensus --log-tfb
   ```

   Key Auto-K switches:

   | Option | Purpose |
   | --- | --- |
   | `--auto-k-method {consensus,silhouette,eigengap,dpmeans,xmeans,gmeans}` | Pick a single heuristic for the whole run. |
   | `--auto-k-method-cycle consensus,silhouette,xmeans` | Cycle through a comma-separated list of heuristics. Combine with `--auto-k-cycle-level {round,epoch}` to change cadence. |
   | `--auto-k-consensus-methods silhouette,dpmeans` | Restrict the voters used by the consensus strategy. |
   | `--auto-k-min-k` / `--auto-k-max-k` | Clamp the admissible cluster-count range. |
   | `--auto-k-dp-lambda` and `--auto-k-dp-max-iter` | Tune the DP-means penalty and refinement steps. |
   | `--auto-k-xmeans-bic-threshold`, `--auto-k-max-depth`, `--auto-k-gmeans-threshold` | Control the behaviour of the X-means and G-means inspired splits. |
   | `--auto-k-sample-size` | Limit the number of embeddings analysed per estimation pass (defaults to 2048). |
   | `--auto-k-log-embeddings` | Store the sampled embeddings and their pseudo labels in TensorBoard under `AutoK/*_embeddings`. |

   You can switch heuristics on the fly by pairing the cycling options with curriculum rounds. For example, `--auto-k-method-cycle silhouette,dpmeans,consensus --auto-k-cycle-level round` evaluates each round with a different estimator while keeping TensorBoard logs aligned.

7. Visualise clustering progress live in TensorBoard by enabling TensorBoard logging (already `True` in `configs/base.yaml`) and running:

   ```bash
   # in a separate terminal
   tensorboard --logdir sessions --port 6006
   ```

   Visit `http://localhost:6006` to monitor metrics such as AN statistics, loss curves, evaluation accuracy, and the new `AutoK/EstimatedK`, `AutoK/Vote_*`, and `AutoK/<method>_*` traces. When `--auto-k-log-embeddings` is enabled, TensorBoard’s **Projector** tab visualises the sampled embeddings colour-coded by the inferred cluster labels.

8. Run on CUDA with manual setup (optional): if you have multiple CUDA toolkits installed, you can control visibility by prefixing the command: `CUDA_VISIBLE_DEVICES=0 python3 main.py ...`. Combine this with `--gpus 0` to ensure PyTorch only initialises the desired devices.

Every time the `main.py` is run, a new session will be started with the name of current timestamp and all the related files will be stored in folder `sessions/timestamp/` including checkpoints, logs, etc.

### Pre-trained model

To play with the pre-trained model, please go to ResNet18 / AlexNet. A few things need to be noticed:

- The model is saved in PyTorch format
- It expects RGB images that their pixel values are normalised with the following mean RGB values `mean=[0.485, 0.456, 0.406]` and std RGB values `std=[0.229, 0.224, 0.225]`. Prior to normalisation the range of the image values must be `[0.0, 1.0]`.
