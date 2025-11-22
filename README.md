<div align="center">
<h1>UniBiomed: A Universal Foundation Model for Grounded Biomedical Image Interpretation</h1>

<a href="https://arxiv.org/abs/2504.21336"><img src='https://img.shields.io/badge/arXiv-Preprint-red' alt='Paper PDF'></a>
<a href='https://huggingface.co/Luffy503/UniBiomed'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>
<a href='https://huggingface.co/datasets/Luffy503/UniBiomed'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-green' alt='Dataset'></a>
</div>

We introduce **UniBiomed**, the first universal foundation model for grounded biomedical image interpretation, which is capable of generating accurate diagnostic findings and simultaneously segmenting the corresponding biomedical targets. UniBiomed is based on a novel integration of Multi-modal Large Language Model (MLLM) and Segment Anything Model (SAM), which can effectively unify diverse biomedical tasks in universal training for advancing grounded interpretation.

![teaser](assets/fig1.png)

[//]: # (## News)

[//]: # ()
[//]: # (- **2025-04-30:** Paper, code, models, and datasets are released.)

## Installation
```bash
git clone https://github.com/Luffy03/UniBiomed
cd UniBiomed
conda create -n UniBiomed python=3.10
conda activate UniBiomed
conda install pytorch==2.3.1 torchvision==0.18.1 pytorch-cuda=12.1 cuda -c pytorch  -c "nvidia/label/cuda-12.1.0" -c "nvidia/label/cuda-12.1.1"
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.3/index.html
pip install -r requirements.txt
```

### Pre-trained Models

You need to download [sam2-hiera-large](https://huggingface.co/facebook/sam2-hiera-large) in the 'pretrained' path.
```
./ # project root
pretrained/
├── sam2_hiera_large.pt
```

## Download Datasets

Our curated datasets are available at [Hugging face](https://huggingface.co/Luffy503/UniBiomed). Some of the datasets should be downloaded and processed from the original links. The datasets are organized as follows:
```
./ # project root
data/Biomed
├── CoCaHis
    ├──train
    ├──train_mask
    ├──test
    ├──test_mask
    ├──train.json
    ├──test.json
├── MedTrinity
├── MSD
├── ...
```

## Usage
Quick start to use our model. A demo script is available at [example.py](https://github.com/Luffy03/UniBiomed/blob/main/example.py) and some examples are placed in './examples'.
```python
import argparse
import torch
from transformers import (AutoModel, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          GenerationConfig)
def parse_args():
    parser = argparse.ArgumentParser(description='UniBiomed')
    parser.add_argument('--model_path', default='Luffy503/UniBiomed')
    return args
args = parse_args()

# load model
model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(
    args.model_path,
    trust_remote_code=True,
)

# define data input, image and text instruction
data_dict = {}
image, text = None, None
data_dict['image'] = image
data_dict['text'] = text

# output
pred_dict = model.predict_forward(**data_dict, tokenizer=tokenizer)
# text description
prediction = pred_dict['prediction']
# segmentation mask
mask = pred_dict['prediction_masks'][0][0]
```

## Training

Run the following command for training (8*H800 GPUs).
```bash
bash tools/dist.sh train projects/unibiomed/configs/biomed.py 8
```

## Evaluation

After training, you need to save hugging face model for evaluation. Replace '$your_model$' as the real model path. The model will be saved to './save_hf'.
```bash
PYTHONPATH=. python projects/unibiomed/hf/convert_to_hf.py projects/unibiomed/configs/biomed.py --pth-model ./work_dirs/biomed/$your_model$.pth --save-path ./save_hf
```

You can use our [trained model](https://huggingface.co/Luffy503/UniBiomed) on hugging face for evaluation.

For segmentation (replace '$datasetname'):
```bash
PYTHONPATH=. python demo/demo_seg2D.py --val_folder /data/Biomed/$datasetname --work-dir ./val_results/$datasetname --model_path Luffy503/UniBiomed

# or one for all
bash demo_seg.sh
```

For grounded disease recognition:
```bash
PYTHONPATH=. python demo/demo_disease.py --data_path ./data/Biomed/Disease/$datasetname --model_path Luffy503/UniBiomed --save_dir ./val_results/Grounded_disease/$datasetname
# eval metrics
python demo/eval_utils/metrics_grounded_disease.py  --root ./data/Biomed/Disease/$datasetname --prediction_dir_path ./val_results/Grounded_disease/$datasetname

# or one for all
bash demo_disease.sh
```

For Region understand:
```bash
PYTHONPATH=. python demo/demo_RegionCap.py --data_path ./data/Biomed/Disease/$datasetname --model_path Luffy503/UniBiomed --save_dir ./val_results/region_understand/$datasetname

# or one for all
bash demo_RegionCap.sh
```

For medtrinity report generation:
```bash
PYTHONPATH=. python demo/demo_Medtrinity.py --model_path Luffy503/UniBiomed
# eval metrics
python demo/eval_utils/metrics_medtrinity.py  --root ./data/Biomed/MedTrinity --gt_json_path train.json --prediction_dir_path ./val_results/MedTrinity
```

For radgenome report generation:
```bash
PYTHONPATH=. python demo/demo_GRG.py --model_path Luffy503/UniBiomed --save_dir ./val_results/Grounded_Report_Generation/RadGenome
# eval metrics
python demo/eval_utils/metrics_grg.py --root ./data/Biomed/RadGenome --prediction_dir_path ./val_results/Grounded_Report_Generation/RadGenome
```

## Acknowledgement <a name="Acknowledgment"></a>

Our work is developed on the great work [Sa2VA](https://github.com/magic-research/Sa2VA). We highly appreciate their great efforts. We also thanks [RadGenome](https://huggingface.co/datasets/RadGenome/RadGenome-ChestCT), [BiomedParse](https://github.com/microsoft/BiomedParse), [VoCo](https://github.com/Luffy03/VoCo), and [MedTrinity](https://github.com/UCSC-VLAA/MedTrinity-25M) for providing data preprocessing toolkits.

## Citation

If you find this repo useful for your research, please consider citing the paper as follows:

```bibtex
@article{wu2025unibiomed,
  title={UniBiomed: A Universal Foundation Model for Grounded Biomedical Image Interpretation},
  author={Wu, Linshan and Nie, Yuxiang and He, Sunan and Zhuang, Jiaxin and Chen, Hao},
  journal={arXiv preprint arXiv:2504.21336},
  year={2025}
}
```
