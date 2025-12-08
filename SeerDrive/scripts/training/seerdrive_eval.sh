# Define
export PYTHONPATH="/scratch/rob535f25s001_class_root/rob535f25s001_class/kushkp/SeerOpt/SeerDrive"
echo "1"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/scratch/rob535f25s001_class_root/rob535f25s001_class/kushkp/SeerOpt/SeerDrive/data/nuplan_maps/maps"
echo "2"
export NAVSIM_EXP_ROOT="/scratch/rob535f25s001_class_root/rob535f25s001_class/kushkp/SeerOpt/SeerDrive/exp"
echo "3"
export NAVSIM_DEVKIT_ROOT="/scratch/rob535f25s001_class_root/rob535f25s001_class/kushkp/SeerOpt/SeerDrive"
echo "4"
export OPENSCENE_DATA_ROOT="/scratch/rob535f25s001_class_root/rob535f25s001_class/kushkp/SeerOpt/SeerDrive/data/navsim"
echo "5"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
echo "6"
CONFIG_NAME=default

### evaluation ###
export CKPT="/scratch/rob535f25s001_class_root/rob535f25s001_class/kushkp/SeerOpt/SeerDrive/ckpts"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
agent=SeerDrive_agent \
agent.checkpoint_path=$CKPT \
agent.config._target_=navsim.agents.SeerDrive.configs.${CONFIG_NAME}.SeerDriveConfig \
experiment_name=eval \
split=test \
scene_filter=navtest