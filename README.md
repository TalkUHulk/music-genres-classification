![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)

# music-genres-classification
## Abstract
The demo is to develop a deep learning model that will identify the genres from music.The model adopted is Inception-ResnetV2 from slim.And the datasat used GTZAN Music Genre Dataset, which is a collection of 1000 songs in 10 genres, is the most widely used dataset.If you wanna train a model by yourself, download it from [GTZAN dataset](http://opihi.cs.uvic.ca/sound/genres.tar.gz).

Below table shows the result on test set:

Accuracy | Value
--------- | ---------
Top-1 | 69.70%
Top-5 | 92.50%

Below picture shows the training-process on tensorboard:
<p align="center">
  <img src="tensorboard/loss.jpg" width="400"> <br />
  <em> Loss </em>
</p>
<p align="center">
  <img src="tensorboard/accuracy.jpg" width="400"> <br />
  <em> Accuracy</em>
</p>

##Pre-Trained Testing

* Step1: download [pre_trained](https://pan.baidu.com/s/1Pg7UH5rj_xCv77Wz4sCy_A) model,and put it into `models/` folder.
* Step2: test by executing the following command:
```python 
python test.py
```

## Training

* Step1: download Dataset GTZAN,and put it into `GTZAN/` folder.
* Step2: create the tfrecords by executing the following command:
```python 
python create_data_to_train.py
```
* Step3: train the model by executing the following command:
```python 
python train.py
```

## Testing

Run the following command to simple test the test-dataset:
```python 
python test.py
```

## References

[1]https://github.com/deepsound-project/genre-recognition/blob/master/README.md



