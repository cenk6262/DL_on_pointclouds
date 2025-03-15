## Installation
We provide a simple bash file to install the environment:

```
git clone --recurse-submodules https://github.com/cenk6262/DL_on_pointclouds.git
cd DL_on_pointclouds
source update.sh
source install.sh
```
Cuda-11.3 is required. Modify the `install.sh` if a different cuda version is used. See [Install](docs/index.md) for detail. 



A short instruction: all experiments follow the simple rule to train and test: 

```
CUDA_VISIBLE_DEVICES=$GPUs python examples/$task_folder/main.py --cfg $cfg $kwargs
```
- $GPUs is the list of GPUs to use, for most experiments (ScanObjectNN, ModelNet40, S3DIS), we only use 1 RTX 4060 (GPUs=0)
- $task_folder is the folder name of the experiment. For example, for s3dis segmentation, $task_folder=s3dis
- $cfg is the path to cfg, for example, s3dis segmentation, $cfg=cfgs/s3dis/pointnext-s.yaml
- $kwargs are the other keyword arguments to use. For example, testing in S3DIS area 5, $kwargs should be `mode=test, --pretrained_path $pretrained_path`. 


Training:
```
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/scanobjectnn/pointnext-s.yaml
```

Trained was on the ScanObjectNN Dataset and in this example we used the Pointnext Model.

With Pointnet++:
```
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/scanobjectnn/pointnet++.yaml
```

With Pointnet:
```
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/scanobjectnn/pointnet.yaml
```


Testing:
```
CUDA_VISIBLE_DEVICES=0
python
examples/classification/main.py
--cfg
cfgs/scanobjectnn/pointnet.yaml
mode=test
--pretrained_path
log/scanobjectnn/scanobjectnn-train-pointnet-ngpus1-seed369-20250209-171239-5MEQCBj2UXnZBBo9JZtxQA/checkpoint/scanobjectnn-train-pointnet-ngpus1-seed369-20250209-171239-5MEQCBj2UXnZBBo9JZtxQA_ckpt_best.pth
```

Where the --cfg and --pretrained_path can be changed acordingly (in this example we used  pointnet.yaml).

Finally add a "data" folder in the root, and insert the ScanObjectNN Dataset as can be donwloaded here: https://hkust-vgd.ust.hk/scanobjectnn/
the "h5_files.zip" has to be downloaded.



