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

    ### Evaluate
    python main_multi.py --evaluate <MODEL-PATH>
    ```
 
 - Train and evaluate CIFAR10 dataset with only two classes
    - Note: Here we split CIFAR10 to two classes, one for animal, another for others
    ```
    ### TrainN
    python main_binary.py

    ### Evaluate
    python main_binary.py --evaluate <MODEL-PATH>
    ```


## For multitask training (TODO)

```
python main_mtl.py --pre-train --alter-train --post-train
```
 