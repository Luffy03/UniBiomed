from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoTokenizer

from xtuner.dataset import ConcatDataset
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.dataset.map_fns import template_map_fn_factory

from third_parts.mmdet.models.losses import DiceLoss, CrossEntropyLoss
from peft import LoraConfig

from projects.unibiomed.models.internvl import InternVL_Slowfast

from projects.unibiomed.models import SAM2TrainRunner, VideoLLaVASAMModel_zero3
from projects.unibiomed.datasets import video_lisa_collate_fn
from projects.unibiomed.models.preprocess.image_resize import DirectResize

from projects.unibiomed.datasets import (ReferSegBiomedDataset, ReferSegBiomedDataset3D, BiomedVQADataset, BiomedRGDataset,
                                         RegionBiomedDataset, RegionMedtrinityDataset, DiseaseBiomedDataset)
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
path = 'OpenGVLab/InternVL2_5-1B'
sam_cfg_path = "sam2_hiera_l.yaml"
sam_ckpt_path = "sam2_hiera_large.pt"
pretrained_pth = None

# Data
template = "phi3_chat"
prompt_template = PROMPT_TEMPLATE.phi3_chat
max_length = 8192

# Scheduler & Optimizer
batch_size = 4  # per_device
accumulative_counts = 4
dataloader_num_workers = 4
max_epochs = 10
optim_type = AdamW
# official 1024 -> 4e-5
lr = 4e-5

betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1  # grad clip
warmup_ratio = 0.05

# Save
save_steps = 500
save_total_limit = 3  # Maximum checkpoints to keep (-1 means unlimited)

special_tokens = ['[SEG]', '<p>', '</p>', '<vp>', '</vp>']

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=path,
    trust_remote_code=True,
    padding_side='right')

extra_image_processor = dict(
    type=DirectResize,
    target_length=1024,
)
#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
model = dict(
    type=VideoLLaVASAMModel_zero3,
    special_tokens=special_tokens,
    frozen_sam2_decoder=False,

    mllm=dict(
        type=InternVL_Slowfast,
        model_path=path,
        freeze_llm=True,
        freeze_visual_encoder=True,
        llm_lora=dict(
            type=LoraConfig,
            r=128,
            lora_alpha=256,
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM'),
        special_tokens=special_tokens,
    ),
    tokenizer=tokenizer,
    grounding_encoder=dict(
        type=SAM2TrainRunner,
        cfg_path=sam_cfg_path,
        ckpt_path=sam_ckpt_path,
    ),
    loss_mask=dict(
        type=CrossEntropyLoss,
        use_sigmoid=True,
        reduction='mean',
        loss_weight=2.0),
    loss_dice=dict(
        type=DiceLoss,
        use_sigmoid=True,
        activate=True,
        reduction='mean',
        naive_dice=True,
        eps=1.0,
        loss_weight=0.5),
    pretrained_pth=pretrained_pth,
    loss_sample_points=True,  # important, if False: the CE will soon convergent
    bs=batch_size,
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################

DATA_ROOT = './data/'

################## Biomed 2d seg
BIOMED_ROOT = DATA_ROOT + 'Biomed/'
BioACDC = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'ACDC',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioBTCV = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'BTCV',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioAMOS_CT = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'amos22/CT',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioAMOS_MRI = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'amos22/MRI',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioBreastUS = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'BreastUS',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioCAMUS = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'CAMUS',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioCDD_CESM = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'CDD-CESM',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioCOVID_19_CT = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'COVID-19_CT',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioCOVID_QU_Ex = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'COVID-QU-Ex',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioCXR = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'CXR_Masks_and_Labels',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioDRIVE = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'DRIVE',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioFH_PS_AOP = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'FH-PS-AOP',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioKiTS2023 = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'KiTS2023',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)

BioAIIB23 = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'AIIB23',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
Bioaorta = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'aorta',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)


BioFlare22 = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'Flare22',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)

BioWORD = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'WORD',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)

BioLGG = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'LGG',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioLIDC_IDRI = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'LIDC-IDRI',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioLiverUS = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'LiverUS',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)

BioMSD_Task01_BrainTumour = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'MSD/Task01_BrainTumour',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioMSD_Task02_Heart = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'MSD/Task02_Heart',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioMSD_Task03_Liver = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'MSD/Task03_Liver',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioMSD_Task04_Hippocampus = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'MSD/Task04_Hippocampus',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioMSD_Task05_Prostate = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'MSD/Task05_Prostate',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioMSD_Task06_Lung = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'MSD/Task06_Lung',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioMSD_Task07_Pancreas = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'MSD/Task07_Pancreas',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioMSD_Task08_HepaticVessel = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'MSD/Task08_HepaticVessel',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioMSD_Task09_Spleen = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'MSD/Task09_Spleen',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioMSD_Task10_Colon = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'MSD/Task10_Colon',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)

BioNeoPolyp = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'NeoPolyp',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioOCT_CME = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'OCT-CME',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioPanNuke = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'PanNuke',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioRadiography_COVID = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'Radiography/COVID',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioRadiography_Lung_Opacity = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'Radiography/Lung_Opacity',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioRadiography_Normal = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'Radiography/Normal',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioRadiography_Viral_Pneumonia = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'Radiography/Viral_Pneumonia',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)

BioREFUGE = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'REFUGE',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
Biosiim_acr_pneumothorax = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'siim-acr-pneumothorax',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioUWaterlooSkinCancer = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'UWaterlooSkinCancer',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)

# Biomed Pathology
BioCoCaHis = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'CoCaHis',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)

BioCryoNuSeg = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'CryoNuSeg',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioDigestPath2019_colonoscopy_tissue_segment = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'DigestPath2019_colonoscopy_tissue_segment',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioGlaS = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'GlaS',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)


BioSICAPv2 = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'SICAPv2',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioWSSS4LUAD = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'WSSS4LUAD',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)

# Radgemone seg
BioRadGenome_seg = dict(
    type=ReferSegBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_ROOT + 'RadGenome',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)

################## Biomed 3d seg
BIOMED_ROOT3D = BIOMED_ROOT + '3D/'
BioMSD_Task06_Lung_3D = dict(
    type=ReferSegBiomedDataset3D,
    image_folder=BIOMED_ROOT3D + 'Task06_Lung',
    expression_file=BIOMED_ROOT3D + 'Task06_Lung' + '/meta_train.json',
    tokenizer=tokenizer,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=8192,
    lazy=True,
    repeats=10,
    special_tokens=['[SEG]'],
    extra_image_processor=extra_image_processor,
    sampled_slices=10,
)
BioMSD_Task09_Spleen_3D = dict(
    type=ReferSegBiomedDataset3D,
    image_folder=BIOMED_ROOT3D + 'Task09_Spleen',
    expression_file=BIOMED_ROOT3D + 'Task09_Spleen' + '/meta_train.json',
    tokenizer=tokenizer,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=8192,
    lazy=True,
    repeats=10,
    special_tokens=['[SEG]'],
    extra_image_processor=extra_image_processor,
    sampled_slices=10,
)
BioMSD_Task10_Colon_3D = dict(
    type=ReferSegBiomedDataset3D,
    image_folder=BIOMED_ROOT3D + 'Task10_Colon',
    expression_file=BIOMED_ROOT3D + 'Task10_Colon' + '/meta_train.json',
    tokenizer=tokenizer,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=8192,
    lazy=True,
    repeats=10,
    special_tokens=['[SEG]'],
    extra_image_processor=extra_image_processor,
    sampled_slices=10,
)

BioCOVID_CT_3D = dict(
    type=ReferSegBiomedDataset3D,
    image_folder=BIOMED_ROOT3D + 'COVID-19_CT',
    expression_file=BIOMED_ROOT3D + 'COVID-19_CT' + '/meta_train.json',
    tokenizer=tokenizer,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=8192,
    lazy=True,
    repeats=10,
    special_tokens=['[SEG]'],
    extra_image_processor=extra_image_processor,
    sampled_slices=10,
)


BIOMED_Disease_ROOT = BIOMED_ROOT + 'Disease/'
### Disease
BioDisease_BrainTumor = dict(
    type=DiseaseBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_Disease_ROOT + 'BrainTumor',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioDisease_ColonCancer = dict(
    type=DiseaseBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_Disease_ROOT + 'ColonCancer',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioDisease_Fibrotic = dict(
    type=DiseaseBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_Disease_ROOT + 'Fibrotic',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioDisease_LiverTumor = dict(
    type=DiseaseBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_Disease_ROOT + 'LiverTumor',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioDisease_LungTumor = dict(
    type=DiseaseBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_Disease_ROOT + 'LungTumor',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioDisease_pneumothorax = dict(
    type=DiseaseBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_Disease_ROOT + 'pneumothorax',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioDisease_ProstateCancer = dict(
    type=DiseaseBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_Disease_ROOT + 'ProstateCancer',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioDisease_Retinal = dict(
    type=DiseaseBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_Disease_ROOT + 'Retinal',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioDisease_BreastTumor = dict(
    type=DiseaseBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_Disease_ROOT + 'BreastTumor',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioDisease_COVID19 = dict(
    type=DiseaseBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_Disease_ROOT + 'COVID19',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioDisease_KidneyTumor = dict(
    type=DiseaseBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_Disease_ROOT + 'KidneyTumor',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioDisease_LungNodule = dict(
    type=DiseaseBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_Disease_ROOT + 'LungNodule',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioDisease_PancreasTumor = dict(
    type=DiseaseBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_Disease_ROOT + 'PancreasTumor',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioDisease_ColonPolyp = dict(
    type=DiseaseBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_Disease_ROOT + 'ColonPolyp',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioDisease_Skin = dict(
    type=DiseaseBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_Disease_ROOT + 'Skin',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
BioDisease_NoFindings = dict(
    type=DiseaseBiomedDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=BIOMED_Disease_ROOT + 'NoFindings',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)


### Region Understand
RegionBioBreastUS = dict(
    type=RegionBiomedDataset,
    tokenizer=tokenizer,
    data_root=BIOMED_ROOT + 'BreastUS',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    max_length=max_length,
)
RegionBioCAMUS = dict(
    type=RegionBiomedDataset,
    tokenizer=tokenizer,
    data_root=BIOMED_ROOT + 'CAMUS',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    max_length=max_length,
)
RegionBioCDD_CESM = dict(
    type=RegionBiomedDataset,
    tokenizer=tokenizer,
    data_root=BIOMED_ROOT + 'CDD-CESM',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    max_length=max_length,
)
RegionBioCOVID_19_CT = dict(
    type=RegionBiomedDataset,
    tokenizer=tokenizer,
    data_root=BIOMED_ROOT + 'COVID-19_CT',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    max_length=max_length,
)
RegionBioCOVID_QU_Ex = dict(
    type=RegionBiomedDataset,
    tokenizer=tokenizer,
    data_root=BIOMED_ROOT + 'COVID-QU-Ex',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    max_length=max_length,
)
RegionBioKiTS2023 = dict(
    type=RegionBiomedDataset,
    tokenizer=tokenizer,
    data_root=BIOMED_ROOT + 'KiTS2023',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    max_length=max_length,
)


RegionBioLGG = dict(
    type=RegionBiomedDataset,
    tokenizer=tokenizer,
    data_root=BIOMED_ROOT + 'LGG',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    max_length=max_length,
)
RegionBioLIDC_IDRI = dict(
    type=RegionBiomedDataset,
    tokenizer=tokenizer,
    data_root=BIOMED_ROOT + 'LIDC-IDRI',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    max_length=max_length,
)
RegionBioMSD_Task03_Liver = dict(
    type=RegionBiomedDataset,
    tokenizer=tokenizer,
    data_root=BIOMED_ROOT + 'MSD/Task03_Liver',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    max_length=max_length,
)

RegionBioMSD_Task06_Lung = dict(
    type=RegionBiomedDataset,
    tokenizer=tokenizer,
    data_root=BIOMED_ROOT + 'MSD/Task06_Lung',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    max_length=max_length,
)
RegionBioMSD_Task07_Pancreas = dict(
    type=RegionBiomedDataset,
    tokenizer=tokenizer,
    data_root=BIOMED_ROOT + 'MSD/Task07_Pancreas',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    max_length=max_length,
)
RegionBioMSD_Task08_HepaticVessel = dict(
    type=RegionBiomedDataset,
    tokenizer=tokenizer,
    data_root=BIOMED_ROOT + 'MSD/Task08_HepaticVessel',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    max_length=max_length,
)
RegionBioMSD_Task10_Colon = dict(
    type=RegionBiomedDataset,
    tokenizer=tokenizer,
    data_root=BIOMED_ROOT + 'MSD/Task10_Colon',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    max_length=max_length,
)

RegionBioNeoPolyp = dict(
    type=RegionBiomedDataset,
    tokenizer=tokenizer,
    data_root=BIOMED_ROOT + 'NeoPolyp',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    max_length=max_length,
)
RegionBioOCT_CME = dict(
    type=RegionBiomedDataset,
    tokenizer=tokenizer,
    data_root=BIOMED_ROOT + 'OCT-CME',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    max_length=max_length,
)
RegionBioPanNuke = dict(
    type=RegionBiomedDataset,
    tokenizer=tokenizer,
    data_root=BIOMED_ROOT + 'PanNuke',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    max_length=max_length,
)
RegionBioRadiography_COVID = dict(
    type=RegionBiomedDataset,
    tokenizer=tokenizer,
    data_root=BIOMED_ROOT + 'Radiography/COVID',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    max_length=max_length,
)
RegionBioRadiography_Lung_Opacity = dict(
    type=RegionBiomedDataset,
    tokenizer=tokenizer,
    data_root=BIOMED_ROOT + 'Radiography/Lung_Opacity',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    max_length=max_length,
)
RegionBiosiim_acr_pneumothorax = dict(
    type=RegionBiomedDataset,
    tokenizer=tokenizer,
    data_root=BIOMED_ROOT + 'siim-acr-pneumothorax',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    max_length=max_length,
)
RegionBioUWaterlooSkinCancer = dict(
    type=RegionBiomedDataset,
    tokenizer=tokenizer,
    data_root=BIOMED_ROOT + 'UWaterlooSkinCancer',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    max_length=max_length,
)

# RegionBiomed Pathology
RegionBioCoCaHis = dict(
    type=RegionBiomedDataset,
    tokenizer=tokenizer,
    data_root=BIOMED_ROOT + 'CoCaHis',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    max_length=max_length,
)

RegionBioCryoNuSeg = dict(
    type=RegionBiomedDataset,
    tokenizer=tokenizer,
    data_root=BIOMED_ROOT + 'CryoNuSeg',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    max_length=max_length,
)

RegionBioGlaS = dict(
    type=RegionBiomedDataset,
    tokenizer=tokenizer,
    data_root=BIOMED_ROOT + 'GlaS',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    max_length=max_length,
)


RegionBioSICAPv2 = dict(
    type=RegionBiomedDataset,
    tokenizer=tokenizer,
    data_root=BIOMED_ROOT + 'SICAPv2',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    max_length=max_length,
)
RegionBioWSSS4LUAD = dict(
    type=RegionBiomedDataset,
    tokenizer=tokenizer,
    data_root=BIOMED_ROOT + 'WSSS4LUAD',
    data_prefix=dict(img_path='train'),
    ann_file='train.json',
    prompt_template=prompt_template,
    max_length=max_length,
)

### RadGemone VQA
BioRadGenome_vqa = dict(
    type=BiomedVQADataset,
    image_folder=BIOMED_ROOT + 'RadGenome',
    json_path=BIOMED_ROOT + 'RadGenome' + '/train_parse.json',
    tokenizer=tokenizer,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=8192,
    lazy=True,
    repeats=2,
    special_tokens=['[SEG]'],
    extra_image_processor=extra_image_processor,
)

### RadGemone RG
BioRadGenome_rg = dict(
    type=BiomedRGDataset,
    image_folder=BIOMED_ROOT + 'RadGenome',
    json_path=BIOMED_ROOT + 'RadGenome' + '/train_parse.json',
    tokenizer=tokenizer,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=8192,
    lazy=True,
    repeats=2,
    special_tokens=['[SEG]'],
    extra_image_processor=extra_image_processor,
)

# Medtrinity
BioMedTrinity = dict(
    type=RegionMedtrinityDataset,
    data_root=BIOMED_ROOT + 'MedTrinity',
    ann_file='train.json',
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    max_length=8192,
)

train_dataset = dict(
    type=ConcatDataset, datasets=[

        # ## Grounded VQA dataset
        BioRadGenome_vqa,
        #
        # ## Grounded RG dataset
        BioRadGenome_rg,

        # Medtrinity
        BioMedTrinity,

        ### biomedseg 2D
        BioACDC, BioAMOS_CT, BioAMOS_MRI, BioBreastUS, BioCAMUS, BioCDD_CESM, BioCOVID_19_CT,
        BioCOVID_QU_Ex, BioCXR, BioDRIVE, BioFH_PS_AOP, BioLGG,
        BioLIDC_IDRI, BioLiverUS,

        BioMSD_Task01_BrainTumour,
        BioMSD_Task02_Heart,
        BioMSD_Task03_Liver,
        BioMSD_Task04_Hippocampus, BioMSD_Task05_Prostate,
        BioMSD_Task06_Lung,
        BioMSD_Task07_Pancreas,
        BioMSD_Task08_HepaticVessel,
        BioMSD_Task09_Spleen,
        BioMSD_Task10_Colon,

        BioNeoPolyp, BioOCT_CME,
        BioRadiography_COVID, BioRadiography_Lung_Opacity, BioRadiography_Normal, BioRadiography_Viral_Pneumonia,
        BioREFUGE, Biosiim_acr_pneumothorax, BioUWaterlooSkinCancer,
        BioPanNuke,BioCoCaHis,
        BioCryoNuSeg, BioDigestPath2019_colonoscopy_tissue_segment,
        BioGlaS, BioSICAPv2,BioWSSS4LUAD,

        ### biomedseg 3D crop as 2D
        BioBTCV, BioWORD,

        BioKiTS2023,
        BioAIIB23,Bioaorta,
        BioFlare22,

        # radgenome seg
        BioRadGenome_seg,

        ### biomedseg 3D
        BioMSD_Task06_Lung_3D,
        BioMSD_Task09_Spleen_3D,
        BioMSD_Task10_Colon_3D,
        BioCOVID_CT_3D,
        
        ### Disease
        BioDisease_BrainTumor,BioDisease_ColonCancer,BioDisease_Fibrotic,
        BioDisease_LiverTumor,BioDisease_LungTumor,BioDisease_pneumothorax,BioDisease_ProstateCancer,BioDisease_Retinal,
        BioDisease_BreastTumor,
        BioDisease_COVID19,
        BioDisease_KidneyTumor,
        BioDisease_LungNodule,
        BioDisease_PancreasTumor,
        BioDisease_ColonPolyp,
        BioDisease_Skin,
        BioDisease_NoFindings,

        ### RegionBiomedseg
        RegionBioBreastUS, RegionBioCDD_CESM, RegionBioCOVID_19_CT,
        RegionBioCOVID_QU_Ex, RegionBioLGG,
        RegionBioLIDC_IDRI,
        RegionBioMSD_Task03_Liver,
        RegionBioMSD_Task06_Lung,
        RegionBioMSD_Task07_Pancreas,
        RegionBioMSD_Task08_HepaticVessel,
        RegionBioMSD_Task10_Colon,

        RegionBioNeoPolyp, RegionBioOCT_CME,
        RegionBioRadiography_COVID, 
        RegionBiosiim_acr_pneumothorax, RegionBioUWaterlooSkinCancer,
        RegionBioKiTS2023,

        RegionBioPanNuke,RegionBioCoCaHis,
        RegionBioCryoNuSeg,
        RegionBioGlaS, RegionBioSICAPv2,RegionBioWSSS4LUAD,

    ]
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=video_lisa_collate_fn)
)

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='bfloat16'
)

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    # dict(type=DatasetInfoHook, tokenizer=tokenizer),
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=False,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
