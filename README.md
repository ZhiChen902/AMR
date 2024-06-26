# Adaptive Multi-step Refinement Network for Robust Point Cloud Registration

PyTorch implementation of the paper:

Adaptive Multi-step Refinement Network for Robust Point Cloud Registration

## Introduction

Point Cloud Registration (PCR) estimates the relative rigid transformation between two point clouds of the same scene. Despite significant progress with learning-based approaches, existing methods still face challenges when the overlapping region between the two point clouds is small. In this paper, we propose an adaptive multi-step refinement network that refines the registration quality at each step by leveraging the information from the preceding step. To achieve this, we introduce a training procedure and a refinement network. Firstly, to adapt the network to the current step, we utilize a generalized one-way attention mechanism, which prioritizes the last step's estimated overlapping region, and we condition the network on step indices. Secondly, instead of training the network to map either random transformations or a fixed pre-trained model's estimations to the ground truth, we train it on transformations with varying registration qualities, ranging from accurate to inaccurate, thereby enhancing the network's adaptiveness and robustness. Despite its conceptual simplicity, our method achieves state-of-the-art performance on both the 3DMatch/3DLoMatch and KITTI benchmarks. Notably, on 3DLoMatch, our method reaches \textbf{80.4\%} recall rate, with an absolute improvement of \textbf{1.2\%}. Our code will be made public upon publication.

## Requirements

If you are using conda, you may configure SC2-PCR++ as:

    conda env create -f environment.yml
    conda activate SC2_PCR
    
## 3DMatch

### Data preparation

Downsample and extract FPFH and FCGF descriptors for each frame of the 3DMatch test dataset. [Here](https://drive.google.com/file/d/1kRwuTHlNPr9siENcEMddCO23Oaq0cz-X/view?usp=sharing) we provide the processed test set with pre-computed FPFH/FCGF descriptors. The data should be organized as follows:

```
--data--3DMatch                
        ├── fragments                 
        │   ├── 7-scene-redkitechen/
        |   |   ├── cloud_bin_0.ply
        |   |   ├── cloud_bin_0_fcgf.npz
        |   |   ├── cloud_bin_0_fpfh.npz
        │   |   └── ...      
        │   ├── sun3d-home_at-home_at_scan1_2013_jan_1/      
        │   └── ...                
        ├── gt_result                   
        │   ├── 7-scene-redkitechen-evaluation/   
        |   |   ├── 3dmatch.log
        |   |   ├── gt.info
        |   |   ├── gt.log
        │   |   └── ...
        │   ├── sun3d-home_at-home_at_scan1_2013_jan_1-evaluation/
        │   └── ...                               
```

### Testing

Use the following command for testing.

```bash
python ./test_3DMatch.py --config_path config_json/config_3DMatch_FPFH.json
```
or
```bash
python ./test_3DMatch.py --config_path config_json/config_3DMatch_FCGF.json
```

The CUDA_DEVICE and basic parameters can be changed in the json file.

```

## Acknowledgements
- [PREDATOR](https://github.com/prs-eth/OverlapPredator)
- [GeoTransformer](https://github.com/qinzheng93/GeoTransformer)

We thank the respective authors for open sourcing their methods, the code is heavily borrowed from [GeoTransformer](https://github.com/qinzheng93/GeoTransformer)
