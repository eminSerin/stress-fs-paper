#!/bin/bash

export SUBJECTS_DIR=...  # Set your SUBJECTS_DIR here.
fsgd_base_dir=...  # Set your fsgd_base_dir here.

for hemi in lh rh; do
  for smoothness in 10; do
    for meas in volume thickness area; do
      for tail in pos abs; do
        glmdir=${SUBJECTS_DIR}/${hemi}.${meas}.${smoothness}_increase.glmdir 
        mri_glmfit-sim \
          --glmdir ${glmdir} \
          --cache 3 ${tail} \
          --cwp 0.05  \
          --2spaces
      done
    done
  done
done
