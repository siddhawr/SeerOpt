SPLIT=test  # SPLIT=trainval

export PYTHONPATH="/scratch/rob535f25s001_class_root/rob535f25s001_class/kushkp/SeerOpt/SeerDrive"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/scratch/rob535f25s001_class_root/rob535f25s001_class/kushkp/SeerOpt/SeerDrive/data/nuplan_maps/maps"
export NAVSIM_EXP_ROOT="/scratch/rob535f25s001_class_root/rob535f25s001_class/kushkp/SeerOpt/SeerDrive/exp"
export NAVSIM_DEVKIT_ROOT="/scratch/rob535f25s001_class_root/rob535f25s001_class/kushkp/SeerOpt/SeerDrive"
export OPENSCENE_DATA_ROOT="/scratch/rob535f25s001_class_root/rob535f25s001_class/kushkp/SeerOpt/SeerDrive/data/navsim"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
split=$SPLIT \
cache.cache_path='/scratch/rob535f25s001_class_root/rob535f25s001_class/kushkp/SeerOpt/SeerDrive/exp/metric_cache' \
scene_filter.frame_interval=1
