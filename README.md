# Transfer-Learning
Inception V3 for Transfer Learning on Cats and Dogs

I have added the weights file and the predict file 

Instructions to use the predict file:
Dependency:
  1. Tensorflow = 1.2.1
  2. Keras = 2+
  
Installing Tensorflow-cpu to run ```pip --no-cache-dir install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.1-cp36-cp36m-linux_x86_64.whl```


On the terminal type


1. For Images saved locally
```python predict.py --image imagename.jpg --model_name inception.model```


2. For Images on the net
```python predict.py --image_url www.imagename.jpg --model_name inception.model```
