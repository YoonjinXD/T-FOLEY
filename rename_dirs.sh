#!/bin/bash

# Base directories
BASE_DIR_DEV="./DCASE_2023_Challenge_Task_7_Dataset/dev"
BASE_DIR_EVAL="./DCASE_2023_Challenge_Task_7_Dataset/eval"

# Function to rename directories
rename_dirs() {
    local base_dir=$1
    mv "${base_dir}/dog_bark" "${base_dir}/DogBark"
    mv "${base_dir}/footstep" "${base_dir}/Footstep"
    mv "${base_dir}/gunshot" "${base_dir}/GunShot"
    mv "${base_dir}/keyboard" "${base_dir}/Keyboard"
    mv "${base_dir}/moving_motor_vehicle" "${base_dir}/MovingMotorVehicle"
    mv "${base_dir}/rain" "${base_dir}/Rain"
    mv "${base_dir}/sneeze_cough" "${base_dir}/Sneeze_Cough"
}

# Rename directories in both dev and eval
rename_dirs "$BASE_DIR_DEV"
rename_dirs "$BASE_DIR_EVAL"

echo "Directories have been renamed in both dev and eval."
