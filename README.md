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
    python main_multi.py --evaluate <MODEL-PATH>
    ```
 
 - Train and evaluate CIFAR10 dataset with only two classes
    - Note: Here we split CIFAR10 to two classes, one for animal, another for others
    ```
    ### Train
    python main_binary.py

    ### Evaluation
    python main_binary.py --evaluate <MODEL-PATH>
    ```


## For multitask training

```
python main_mtl.py --pre-train --alter-train --post-train
```
 
## Result

| Model             | Acc.        |
| ----------------- | ----------- |
| Multi-Class       | 93.99%      |
| Binary            | 98.96%      |
| ----------------- | ----------- |
| Multi-Task        | 92.76% & 98.52% |