#!/bin/bash

#---load modules---
ml load GCC
ml load Miniconda3

#---inputs---
idx_run=${1:-19}
rootdirMuSAruns=${2:-"/kyukon/data/gent/vo/000/gvo00090/vsc44965/Doctoraat/Python/Machine_learning/Traditional_MLA/sd/MuSA/test/"}
model_only_sites=${3:-True}

pythonpath="/kyukon/data/gent/vo/000/gvo00090/\
vsc44965/Conda/envs/MuSAenv/bin/python"


#---preprocess---
cfg_path=$($pythonpath preprocessMuSArunTile.py --date_ini "2015-09-01 00:00" \
                                            --date_end "2024-08-31 23:00" \
                                            --rootdirMuSAruns "$rootdirMuSAruns" \
                                            --model_only_sites "$model_only_sites" \
                                            --idx_tile "$idx_run"
                                            )

#---run MuSA--- -> also exports the results to another format
export MUSA_CONFIG=$cfg_path
${pythonpath} runMuSAtile.py 

#---remove the config file after the run---
rm -f $cfg_path