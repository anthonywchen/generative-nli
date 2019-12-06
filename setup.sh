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

# ##########################################
### Downloads data into raw data directory
##########################################
mkdir -p raw_data
cd raw_data/

#### DATASETS FOR TRAINING

# Download SNLI
wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
unzip snli_1.0.zip
rm -r __MACOSX snli_1.0.zip
mv snli_1.0 snli

# Download MNLI
wget https://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip
unzip multinli_1.0.zip
rm -r __MACOSX multinli_1.0.zip
mv multinli_1.0 mnli

#### DATASETS FOR EVALUATING GENERALIZATION

# Download RTE
wget https://dl.fbaipublicfiles.com/glue/superglue/data/v2/RTE.zip
unzip RTE.zip
rm RTE.zip
mv RTE rte

# Download SCITAIL

#### ADVERSARIAL DATASETS FOR EVALUATING GENERALIZATION

# Download Adversarial NLI
wget https://dl.fbaipublicfiles.com/anli/anli_v0.1.zip
unzip anli_v0.1.zip
rm -r __MACOSX anli_v0.1.zip
mv anli_v0.1 anli

# Download HANS
mkdir hans
cd hans
wget https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt
cd ../