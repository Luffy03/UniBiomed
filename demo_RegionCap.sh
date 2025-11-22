#!/bin/bash

DATA_LIST=("RadGenome"
'CoCaHis' 'CRAG' 'CryoNuSeg' 'GlaS' 'MoNuSeg' 'PanNuke' 'SICAPv2' 'WSSS4LUAD'
'KiTS2023'
'CXR_Masks_and_Labels' 'Radiography/COVID' 'Radiography/Lung_Opacity' 'Radiography/Normal' 'Radiography/Viral_Pneumonia' 'COVID-QU-Ex' 'CDD-CESM' 'siim-acr-pneumothorax'
'BreastUS' 'LiverUS' 'CAMUS' 'FH-PS-AOP'
'REFUGE' 'DRIVE' 'UWaterlooSkinCancer' 'NeoPolyp' 'OCT-CME'
'CHAOS' '3Dircadb1' 'MSD/Task03_Liver' 'MSD/Task06_Lung' 'MSD/Task07_Pancreas' 'MSD/Task09_Spleen' 'MSD/Task10_Colon'
           'LIDC-IDRI' 'SLIVER07' 'COVID-19_CT'
           'TCIAPancreas' 'aorta'
           'LGG' 'MSD/Task02_Heart' 'MSD/Task04_Hippocampus' 'MSD/Task05_Prostate' 'MSD/Task08_HepaticVessel'
           'ACDC'
           'MSD/Task01_BrainTumour' 'BTCV' 'WORD' 'Flare22' 'KiTS2023' 'amos22/CT' 'amos22/MRI')

for name in "${DATA_LIST[@]}"; do
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python demo/demo_RegionCap.py --data_path ./data/Biomed/$name/test --annotation_file ./data/Biomed/$name/test.json --save_dir ./val_results/Region_Understand/$name --model_path ./save_hf_region
done