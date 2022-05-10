# BERT-based ChatBot

## Introduction
+ BERT 기반의 ChatBot 모델
+ 코사인 유사도를 통해서 문장의 유사도를 비교하여 알맞은 대답을 출력함

## Data
+ AI Hub의 개방 데이터를 이용
    + 감성대와 말뭉치([Link](https://aihub.or.kr/aidata/7978))
+ 해당 데이터에서 Q&A 데이터를 정제하여 사용함
    + [PreProcessing Code](https://github.com/JoSangYeon/BERT-based_ChatBot/blob/master/data/Pre-Processing.ipynb)

## Model
+ BERT : Sentence-BERT
+ pre-Training : https://huggingface.co/jhgan/ko-sroberta-multitask
  + KorSTS, KorNLI
    + Cosine Pearson: 84.77
    + Cosine Spearman: 85.60
    + Euclidean Pearson: 83.71
    + Euclidean Spearman: 84.40
    + Manhattan Pearson: 83.70
    + Manhattan Spearman: 84.38
    + Dot Pearson: 82.42
    + Dot Spearman: 82.33

## Result

### Performance table

## Conclusion

## How to usage