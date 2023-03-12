# Auto-Multitask-Learning on CIFAR10

## Environment
You can build on your conda environment from the provided environment.yml. Feel free to change the env name in the file.
```
conda env create -f environment.yml
```

## For single task training

 - Train and evaluate CIFAR10 dataset with all 10 classes
    ```
    ### Train
    python main_multi.py

    ### Evaluation
    python main_multi.py --evaluate <CKPT-PATH>
    ```
 
 - Train and evaluate CIFAR10 dataset with only two classes
    - Note: Here we split CIFAR10 to two classes, one for animal, another for others
    ```
    ### Train
    python main_binary.py

    ### Evaluation
    python main_binary.py --evaluate <CKPT-PATH>
    ```


## For multitask training

```
### Train (force first 5 layers to be shared for the two tasks. This can be changed.)
python main_mtl.py --pre-train --alter-train --post-train --shared 5

### Evaluation
python main_mtl.py --evaluate <CKPT-PATH> --visualize
```
 
## Result

Checkpoints can be downloaded [here](https://drive.google.com/drive/folders/1NJlXeACgj_geiC6xn9JdZ_mBD9qrN6Od?usp=share_link).

### Single Model

| Model             | Acc.        |
| ----------------- | ----------- |
| Multi-Class       | 93.99%      |
| Binary            | 98.96%      |


### Multi-task Model

| Model                  | Multi-class Acc. | Binary Acc. |
| ---------------------  | ---------------- | ----------- |
| First 0 layers shared  | 90.50%           | 98.09%      |
| First 5 layers shared  | 92.53%           | 98.32%      |
| First 15 layers shared | 90.34%           | 98.34%      |


## Reference

- [MobileNet Model](https://github.com/kuangliu/pytorch-cifar)
- [AutoMTL paper and code](https://github.com/zhanglijun95/AutoMTL)