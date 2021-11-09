import os
import sys
import shutil
import pandas as pd
import utils
import configparser as ConfigParser

# Read in config file
args        = utils.parse_args()
config_name = args['config']
config      = ConfigParser.ConfigParser(allow_no_value=True)
config.read(f"configs/{config_name}")

quiet = config.getboolean('DEFAULT', 'quiet')
early_stopping = config.getboolean('training', 'early_stopping')

# -----------------------------------------------------------------------------
# Extracting Path of Best model
# Load training csv
path_supliment = config['data']['augment']
if path_supliment in ['True', 'False']:
    path_supliment=''
else:
    path_supliment+='/'
csv_path = config['output']['directory'] +'/'+ path_supliment + config['output']['training_evaluation']
df = pd.read_csv(csv_path)
best = df.iloc[list(df['validation_update'])].iloc[-1]
best_epoch = int(best.name)
path_supliment = config['data']['augment']
path_supliment = path_supliment+'/' if path_supliment not in ['False', 'false'] else ''
MODEL_PATH = config['output']['directory'] +'/'+ path_supliment + str(best_epoch) + '.pt'
MODEL_EVAL_PATH = config['output']['directory'] +'/'+ path_supliment

# -----------------------------------------------------------------------------
#
OUT_PATH = MODEL_EVAL_PATH[:7] + 'reduced/' + MODEL_EVAL_PATH[7:] #-file name
files = os.listdir(MODEL_EVAL_PATH)

os.makedirs(OUT_PATH, exist_ok=True)
tmp = shutil.copy(MODEL_PATH, OUT_PATH)
for f in files:
    if '.pt' not in f:
        if not (('tmp' in f) and ('.png' in f)) or ('.npz' in f): # Uncomment to ignore MP4 images if I like
            tmp = shutil.copy(MODEL_EVAL_PATH+f, OUT_PATH)
