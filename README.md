# Transfer-Learning
Inception V3 for Transfer Learning on Cats and Dogs

File to train the Inception v3 model on any dataset is added(```transfer.py```)

I have added the weights file and the predict file(```predict.py```)

## Instructions to use the predict file:

Recommended to use [**Anaconda 3**](https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh)

Dependencies:
  1. Tensorflow = 1.2.1
  2. Keras = 2+
  
Installing Tensorflow-cpu :
```pip install tensorflow```

Installing keras:
```pip install keras```

### Project file structure
```
Transfer-Learning
| inception.model 
| README.md
| predict.py
| transfer.py
|__test_set
|   |__cats
|   |    cat01.jpg
|   |    cat02.jpg
|   |    ...
|   |__dogs
|   |    dog01.jpg
|   |    dog02.jpg
|   |    ..
|
|
|__training_set
|   |__cats
|   |    cat01.jpg
|   |    cat02.jpg
|   |    ...
|   |__dogs
|   |    dog01.jpg
|   |    dog02.jpg
|   |    ..
 ```

### Operations with transfer.py

1. Add the files in **training_set** and **test_set**

2. Run ```python transfer.py --nb_epoch 5 --batch_size 320 --plot --output_model_file filename.model ```

3. Later use the saved weights to predict any Image of cat or dog from the Internet or saved Images.

### Operations for predict.py

1. For Images saved locally
```python predict.py --image imagename.jpg --model_name inception.model```


2. For Images on the net
```python predict.py --image_url www.imagename.jpg --model_name inception.model```


