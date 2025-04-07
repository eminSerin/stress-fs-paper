#!/bin/bash

export SUBJECTS_DIR=...  # Set your SUBJECTS_DIR here.
fsgd_base_dir=...  # Set your fsgd_base_dir here.
fsgd_file=${fsgd_base_dir}/fsgd_increase.fsgd

for hemi in lh rh; do
  for smoothness in 10; do
    for meas in volume thickness area; do
      mri_glmfit \
        --y ${hemi}.${meas}.${smoothness}.mgh \
        --fsgd ${fsgd_file} \
        --C ${fsgd_base_dir}/contrast_average.mtx \
        --C ${fsgd_base_dir}/contrast_men\>women.mtx \
        --C ${fsgd_base_dir}/contrast_women\>men.mtx \
        --C ${fsgd_base_dir}/contrast_men.mtx \
        --C ${fsgd_base_dir}/contrast_women.mtx \
        --surf fsaverage ${hemi} \
        --cortex \
        --glmdir ${hemi}.${meas}.${smoothness}_increase.glmdir
    done
  done
done