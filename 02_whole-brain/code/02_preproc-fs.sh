#!/bin/bash

export SUBJECTS_DIR=...  # Set your SUBJECTS_DIR here.
fsgd_base_dir=...  # Set your fsgd_base_dir here.
fsgd_file=${fsgd_base_dir}/fsgd_increase.tsv

for hemi in lh rh; do
  for smoothing in 10; do
    for meas in volume thickness area; do
      mris_preproc --fsgd ${fsgd_file} \
        --cache-in ${meas}.fwhm${smoothing}.fsaverage \
        --target fsaverage \
        --hemi ${hemi} \
        --out ${SUBJECTS_DIR}/${hemi}.${meas}.${smoothing}.mgh
    done
  done
done