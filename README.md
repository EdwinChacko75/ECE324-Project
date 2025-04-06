# Install Git LFS
sudo apt update
sudo apt install git-lfs

# Initialize Git LFS
git lfs install

cd reasonix

# Set up the enviorment
pip install -r requirements.txt

# Add Llama3.2 Keys from hft
huggingface-cli login

# Run stuff
/home/ubuntu/.local/bin/torchrun --nproc_per_node=2 main.py --task=reward