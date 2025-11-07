# Environment Setup 
## Install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
source $HOME/miniconda/bin/activate
conda init source ~/.bashrc

## Install PyTorch and other libraries
conda create -y -n iml-proj python=3.10
conda activate iml-proj
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate pandas

## Install OCaml to run local tests
sudo apt-get update && sudo apt-get install -y ocaml

# Usage
python baseline.py --model_id 'model name'