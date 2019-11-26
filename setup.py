### Run in root directory of repository ###

#########################################
### Sets up a virtualenv, activates the env, and installs pip packages
##########################################
virtualenv -p python3 answer-generation-env
source generative-nli-env/bin/activate

# Install the all pip packages
pip install -r requirements.txt

##########################################
### Downloads the data into the raw directory
##########################################
mkdir -p raw_data
cd raw_data/