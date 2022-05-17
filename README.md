# BERT-based ChatBot

## Introduction
+ BERT 기반의 ChatBot 모델
+ 코사인 유사도를 통해서 문장의 유사도를 비교하여 알맞은 대답을 출력함

## Data
+ AI Hub의 개방 데이터를 이용
    + 감성대와 말뭉치([Link](https://aihub.or.kr/aidata/7978))
+ 해당 데이터에서 Q&A 데이터를 정제하여 사용함
    + [PreProcessing Code](https://github.com/JoSangYeon/BERT-based_ChatBot/blob/master/data/Pre-Processing.ipynb)
+ Data Shape

|    | Q                                                                            | A                                                                      |
|---:|:-----------------------------------------------------------------------------|:-----------------------------------------------------------------------|
|  0 | 아내가 드디어 출산하게 되어서 정말 신이 나.                                  | 아내분이 출산을 하시는군요. 정말 축하드려요.                           |
|  1 | 당뇨랑 합병증 때문에 먹어야 할 약이 열 가지가 넘어가니까 스트레스야.         | 약 종류가 많아 번거로우시겠어요.                                       |
|  2 | 고등학교에 올라오니 중학교 때보다 수업이 갑자기 어려워져서 당황스러워.       | 고등학교 수업이 중학교 때와는 다르게 갑자기 어려워서 당황스러우시군요. |
|  3 | 재취업이 돼서 받게 된 첫 월급으로 온 가족이 외식을 할 예정이야. 너무 행복해. | 재취업 후 첫 월급이라 정말 기쁘시겠어요.                               |
|  4 | 빚을 드디어 다 갚게 되어서 이제야 안도감이 들어.                             | 기분 좋으시겠어요. 앞으로는 어떻게 하실 생각이신가요?                  |


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
```python main.py```
```shell
USER >>> 안녕하세요?
 BOT >>> 안녕하세요.
	유사도 : 93.9380%
	추론 시간 : 4.124992609024048
USER >>> 만나서 반가워요
 BOT >>> 반갑습니다.
	유사도 : 92.3732%
	추론 시간 : 1.0769999027252197
USER >>> 오늘 날씨가 정말 좋네요!
 BOT >>> 하늘을 보고 웃어보세요.
	유사도 : 89.7211%
	추론 시간 : 1.077988862991333
USER >>> ㅎㅎ 네~
 BOT >>> 지쳤나봐요.
	유사도 : 88.8440%
	추론 시간 : 1.085996389389038
USER >>> exit
```

### Performance table

## Conclusion

## How to usage
1. If this is the default setting
    + ```git clone https://github.com/JoSangYeon/BERT-based_ChatBot.git```
    + you can use it immediately by running main after "git clone".
    + ```python main.py```


2. If you change the data that the chatbot uses<br>
   + preprocess the data according to the format of 'QA_data.csv' mentioned above
   + ```Pre-Processing Your Custom Datasets```