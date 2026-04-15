# EMASAM

This repository contains reproduction code for the research paper titled **"EMASAM: a Computationally Efficient Sharpness-Aware Minimization via EMA-Guided Perturbations"**.  
Exponential Moving Average Sharpness-Aware Minimization (EMASAM) is a computationally efficient alternative to SAM. Unlike SAM, EMASAM does not depend on loss-gradient information during the perturbation step. Instead, it determines the perturbation direction based on the difference between the current model parameters and those of an EMA-based shadow model, which serves as a stable reference.
By driving the parameters away from a temporally smoothed reference toward a less stable region, the resulting perturbation acts as an efficient approximation for the worst-case direction.

The paper has been accepted at the International Conference on Pattern Recognition (ICPR) 2026
Project Page: http://www.vip.sc.e.titech.ac.jp/proj/AACE/AACE.html  

## Training
Use this commands to train a model.  

Standard training:
```bash
python train.py --config <PATH_TO_CONFIG_FILE> --log_dir <RESULT_DIRECTORY> --log_name <LOG_NAME>
```

## Citation
Tanapat Ratchatorn and Masayuki Tanaka, **“EMASAM: a Computationally Efficient Sharpness-Aware Minimization via EMA-Guided Perturbations”**, International Conference on Pattern Recognition (ICPR), August, 2026.
```bash
@INPROCEEDINGS{10647582,
  author={Ratchatorn, Tanapat and Tanaka, Masayuki},
  booktitle={2026 International Conference on Pattern Recognition (ICPR)}, 
  title={EMASAM: a Computationally Efficient Sharpness-Aware Minimization via EMA-Guided Perturbations}, 
  year={2026}}
