# Transfer-Learning
Inception V3 for Transfer Learning on Cats and Dogs

File to train the Inception v3 model on any dataset is added(```transfer.py```)
I have added the weights file and the predict file(```predict.py```)

## Instructions to use the predict file:
Dependencies:
  1. Tensorflow = 1.2.1
  2. Keras = 2+
  
Installing Tensorflow-cpu :
```pip --no-cache-dir install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.1-cp36-cp36m-linux_x86_64.whl```

Installing keras:
```pip install keras```


### Operations with transfer.py

1. Add the files in *training_set* and *test_set*

2. Run ```python transfer.py --nb_epoch 5 --batch_size 320 --plot --output_model_file filename.model ```

3. Later use the saved weights to predict any Image of cat or dog from the Internet or saved Images.

### Operations for predict.py

1. For Images saved locally
```python predict.py --image imagename.jpg --model_name inception.model```


2. For Images on the net
```python predict.py --image_url www.imagename.jpg --model_name inception.model```


