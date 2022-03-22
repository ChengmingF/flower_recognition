# flower_recognition

This is the project code of course Machine Learning, a Bayesian Perspective (TUdelft).

Usage: \
1.  Download the dataset from https://www.kaggle.com/datasets/alxmamaev/flowers-recognition, and extract the dataset folder 'flowers' into the project folder. \

2.  Run train.py:\
    ```
      python train.py --net vgg16 --epochs 30 --new_fc True 
    ```
    For help (suggest run this first to check what parameters you want to set) \
    ```
      python train.py --h 
    ```
3.  The tensorboard are saved in folder 'log', please check the evalutations in tensorbard.\

4.  The weight are saved in folder 'weights'. \

5.  Train.sh is for running a loop with different network and layers. \
    command:  
    ```
    sh train.sh
    ```
