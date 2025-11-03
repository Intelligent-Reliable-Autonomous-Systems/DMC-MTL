## DMC-MTL: Hybrid Recurrent Models for Accurate Crop State Predictions
Developed by Will Solow and Sandhya Saisubramanian
Oregon State University, Corvallis, OR

Correspondance to Will Solow, soloww@oregonstate.edu 

### Installation

1. Create a virtual environment 
* `conda create -n pcalib python=3.12`
* `conda activate pcalib`

2. Clone this repository
* `git clone git@github.com:wsolow/param-calib.git`

3. Install dependencies with pip
* `pip install -e model_engine`
* `pip install numpy pandas omegaconf matplotlib requests tensorboard huggingface_hub` 

### Running experiments
1. See available data in _data/processed_data/
2. Configure train.yaml file in _train_configs/
3. Run experiment with `python3 -m trainers.train_model --config train.yaml`