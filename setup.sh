#** Run in root directory of repository **#

#########################################
### Sets up a virtualenv and installs Python packages
##########################################
virtualenv -p python3 generative-nli-env
source generative-nli-env/bin/activate

# Install the CUDA10.0 version of torch-1.2 since we have CUDA10.0 installed
pip install https://download.pytorch.org/whl/cu100/torch-1.2.0-cp36-cp36m-manylinux1_x86_64.whl
pip install https://download.pytorch.org/whl/cu100/torchvision-0.4.0-cp36-cp36m-manylinux1_x86_64.whl

# Install the all pip packages
pip install -r requirements.txt

# Install APEX
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../

##########################################
### Downloads data into raw data directory
##########################################
mkdir -p raw_data
cd raw_data/

#### DATASETS FOR TRAINING

# Download SNLI

# Download MNLI

#### DATASETS FOR EVALUATING GENERALIZATION

# Download QNLI

# Download RTE

# Download SCITAIL

#### ADVERSARIAL DATASETS FOR EVALUATING GENERALIZATION

# Download Adversarial NLI

# Download HANS
