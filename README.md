# pklGzMakerForTheano

theano나 tensorflow 등 각종 DNN 관련 tutorial에서 예시로 쓰이는 mnist.pkl.gz파일의 format대로 만드는 file과, test file인 CNN으로 구성되어 있습니다.
This project is about making a file format like 'mnist.pkl.gz' file that used for an example of DNN tutorial, such as theano, tensorflow. and have test code that is about Convolutional Neural Network.

예시 로 사용한 dataset은 AVV(Audio Visual Vehicle) dataset입니다. class를 5개로 나누어 사용하였습니다.

the example dataset is AVV(Audio Visual Vehicle) dataset, 5 classes.

classes / training set / validation set / test set

![image](https://cloud.githubusercontent.com/assets/7467605/16002837/9892b5c6-3194-11e6-862f-02576f3cadac.png)

## 사용 방법 (Usage)

*  같은 디렉토리에 해당 db의 data를 넣습니다.
*  폴더별로 labelling됩니다.
*  코드에 들어가서 폴더를 지정합니다.
*  스크립트를 실행합니다.

English
*  put your database files in data directory
*  the files are labled by folders.
*  modify folder name in repklgz.py code 
*  execute repklgz.py script
python repklgz.py


 
## 남은 할 일 (TODO)

* labeling하는 방법 여러가지 만들어보기
* 만들어진 파일 test모듈만들기


## References

* CNN code 출처: https://github.com/lisa-lab/DeepLearningTutorials

* 참고 : http://stackoverflow.com/questions/26107927/how-to-put-my-dataset-in-a-pkl-file-in-the-exact-format-and-data-structure-used

* dataset 출처: http://visionlab.engr.ccny.cuny.edu/~tao/dataset/AudioVisualVehicleDataset.htm
