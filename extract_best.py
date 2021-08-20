import os
import sys
import shutil
import pandas as pd

# Read in config file
args        = utils.parse_args()
config_name = args['config']
config      = ConfigParser.ConfigParser(allow_no_value=True)
config.read(f"configs/{config_name}")

quiet = config.getboolean('DEFAULT', 'quiet')
early_stopping = config.getboolean('training', 'early_stopping')

# -----------------------------------------------------------------------------
# Extracting Path of Best model
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
tmp = shutil.copy(OUT_PATH+str(best_epoch)+'.pt'
for f in files:
    if '.pt' not in f:
        tmp = shutil.copy(MODEL_EVAL_PATH+f, OUT_PATH)
