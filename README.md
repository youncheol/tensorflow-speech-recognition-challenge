# TensorFlow Speech Recognition Challenge

캐글(Kaggle) [**TensorFlow Speech Recognition Challenge**](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge) 프로젝트 ([슬라이드](https://github.com/youncheol/kaggle-tf-speech-recognition-challenge/blob/master/tf-speech-recognition-challenge.pdf))



## Team 

* **Team Name**: 그데목 (Geudemog)

* **Team Member**: [DaMach0](https://github.com/DaMacho), [Nhol](https://github.com/Nhol), [kjw1oo](https://github.com/kjw1oo), [yanggyu17](https://github.com/yanggyu17)



## Code Description

* `preprocessing.py`: 데이터 어그멘테이션, 스펙트럼 변환, TFRecord 저장
* `model.py`: 트레이닝 모델(CNN+LSTM, [DenseNet](https://arxiv.org/abs/1608.06993)) 
* `train.py`: 모델 트레이닝, 저장
* `predict.py`: 모델 불러오기, 라벨 예측



## Result

* Test accuracy : **0.87828**
* Leaderboard position : **132**/1315
