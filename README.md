# EMASAM: a Computationally Efficient Sharpness-Aware Minimization via EMA-Guided Perturbations

This repository contains reproduction code for the research paper titled **"EMASAM: a Computationally Efficient Sharpness-Aware Minimization via EMA-Guided Perturbations"**.  
Exponential Moving Average Sharpness-Aware Minimization (EMASAM) is a computationally efficient alternative to SAM. Unlike SAM, EMASAM does not depend on loss-gradient information during the perturbation step. Instead, it determines the perturbation direction based on the difference between the current model parameters and those of an EMA-based shadow model, which serves as a stable reference.
By driving the parameters away from a temporally smoothed reference toward a less stable region, the resulting perturbation acts as an efficient approximation for the worst-case direction.

The paper has been accepted at the International Conference on Pattern Recognition (ICPR) 2026

Project Page: https://www.vip.sc.eng.isct.ac.jp/proj/EMASAM

## Dependencies Installation
Run this command to install required packages.
```bash
pip install -r requirements.txt
```

## Training
Use this command to train a model from scratch using the EMASAM method.  

EMASAM training:
```bash
python train.py --config <PATH_TO_CONFIG_FILE> --log_dir <RESULT_DIRECTORY> --log_name <LOG_NAME>
```
For additional details on all parameters, please see [train.py](train.py)

## Testing
To obtain the results as shown in the paper, run the following command for inferencing using a pre-trained model trained by EMASAM.
```bash
python test.py --model <MODEL_NAME> --dataset <DATASET_NAME>
```
The script automatically downloads the corresponding pre-trained checkpoint to **"./pretrained_weight/EMASAM_<MODEL_NAME>_<DATASET_NAME>.pth"** and uses it for evaluation.

The checkpoints can also be manually download [HERE](https://drive.google.com/drive/folders/1VfdF6WhEUyWZ1HvOlRui9mPhb0HS2Uza)


**MODEL_NAME**: "resnet18", "resnet50", "wideresnet", "pyramidnet"

**DATASET_NAME**: "cifar10", "cifar100", "fashionmnist", "emnist"

Note: For ImageNet-1k experiments, use DATASET_NAME: "imagenet_1k" and MODEL_NAME: "resnet50", "vit_b_16"

For additional details on all parameters, please see [test.py](test.py)

## Citation
Tanapat Ratchatorn and Masayuki Tanaka, **“EMASAM: a Computationally Efficient Sharpness-Aware Minimization via EMA-Guided Perturbations”**, International Conference on Pattern Recognition (ICPR), August, 2026.
```bash
@INPROCEEDINGS{10647582,
  author={Ratchatorn, Tanapat and Tanaka, Masayuki},
  booktitle={2026 International Conference on Pattern Recognition (ICPR)}, 
  title={EMASAM: a Computationally Efficient Sharpness-Aware Minimization via EMA-Guided Perturbations}, 
  year={2026}}
