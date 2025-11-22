#!/bin/bash

DATA_LIST=('CoCaHis' 'CRAG' 'CryoNuSeg' 'GlaS' 'MoNuSeg' 'PanNuke' 'SICAPv2' 'WSSS4LUAD'
'CXR_Masks_and_Labels' 'Radiography/COVID' 'Radiography/Lung_Opacity' 'Radiography/Normal' 'Radiography/Viral_Pneumonia'
'COVID-QU-Ex' 'CDD-CESM' 'siim-acr-pneumothorax'
'BreastUS' 'LiverUS' 'CAMUS' 'FH-PS-AOP' 'REFUGE' 'DRIVE' 'UWaterlooSkinCancer' 'NeoPolyp' 'OCT-CME'
'CHAOS' '3Dircadb1' 'TCIAPancreas' 'aorta' 'MSD/Task06_Lung' 'MSD/Task09_Spleen' 'MSD/Task10_Colon' 'MSD/Task03_Liver'  'MSD/Task07_Pancreas'
           'LIDC-IDRI' 'SLIVER07' 'AIIB23' 'COVID-19_CT'
           'LGG' 'MSD/Task02_Heart' 'MSD/Task04_Hippocampus' 'MSD/Task05_Prostate' 'MSD/Task08_HepaticVessel'
           'ACDC'
            'BTCV' 'WORD' 'Flare22' 'KiTS2023'
           'amos22/CT' 'amos22/MRI' 'MSD/Task01_BrainTumour')

# DATA_LIST=('siim-acr-pneumothorax' 'NeoPolyp' 'OCT-CME' 'REFUGE' 'MSD/Task10_Colon' 'COVID-19_CT' 'PanNuke' 'LIDC-IDRI' 'amos22/MRI' 'ACDC' 'amos22/CT')

for name in "${DATA_LIST[@]}"; do
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python demo/demo_seg2D.py --val_json test.json --val_folder /data/Biomed/$name --work-dir ./val_results/2D/$name --model_path ./save_hf
done