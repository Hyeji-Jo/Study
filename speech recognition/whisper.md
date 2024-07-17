https://github.com/openai/whisper?tab=readme-ov-file

# 1. 소개
- Whisper가 등장하기 이전 가장 대표적인 음성 인식 모델은 wav2vec
- 2022년 09월 Open AI에서 개발한 자동음성인식(ASR) 모델
- 영어 음성인식 뿐만 아니라 다국어 인식까지 범위를 확장
## 1) wav2vec
- facebook에서 발표한 음성 인식 모델
- 자기 지도 음성인식 모델
- pre-train : 비지도학습, 60,000 시간의 대규모 음성 데이터
- fine-tuing : 적은 수의 labeled 데이터로 지도 학습 진행
- ReLU와 GELU 활성화 함수 사용
- 단점
  - fine-tuning 자체가 복잡
  - fine-tuning 데이터셋을 구성하는 과정이 성능을 좌지우지 할 가능성 높음
  - 특정 데이터에서만 잘 작동하는 '괴짜 모델'이 될 가능성 높음
 
# 2. 데이터
- 데이터 셋 : 680,000 시간(117,000시간 - 96개의 언어 음성/ 125,000시간 - 이를 영어로 번역한 데이터/ 8,000시간의 한국어 데이터)
- label 여부 : weakly-supervised, 준지도 학습
  - label이 지정된 sample과 label이 지정되지 않은 sample의 조합이 training에 사용되는 학습 방법
- fine-tuning 하지 않음
- 데이터 전처리
  - seq2seq 모델이 표준화를 거치지 않은 원래의 텍스트와 오디오의 표현을 충분히 학습하기 위해 거의 진행하지 않음
  - ITN (inverse text normalization)을 거치지 않아서 음성 인식 파이프라인을 단순화 시키기 위해

# 3. 모델
- 모델 구조는 transformer의 인코더-디코더 구조를 그대로 차용
- 일반적으로 하나의 목적에 맞게 fine-tuning되는 모델들은 동시에 여러 task를 수행하지 못함
  - task 종류 : voice activity detection(음성 활동 감지), speaker diarization(화자 분할), inverse text normalization(음성 텍스트로 변환)
  - whisper에서는 이를 동시에 처리하기 위해 task와 conditioning information(조건부 정보)를 decoder의 input에 시퀀스 정보로 제공

# 4. 실험
- whisper는 일반적인 모델들처럼 train-test set을 분리시켜 모델 평가를 하지 않음
- 기존에 공개된 음성 처리 데이터들에 대해 zero-shot setting 에서 모델 평가 진행
## zero-shot
- 모델이 학습 과정에서 배우지 않은 작업을 수행하는 것
  - ex) 유인나 목소리로 음성을 생성하도록 학습한 모델이 예시 샘플을 이용하여 아이유의 목소리로도 음성을 생성하는 것
  - 셰익스피어처럼 글을 쓰도록 학습한 자연어 생성 모델이 마크 트웨인의 스타일로 글을 쓰는 것
- 장점
  - 데이터 부족 문제 해결
  - 유연성 : 다양한 상황이나 분야에 대해 하나의 모델로 대응할 수 있어 모델의 유연성 증가
  - 시간 및 비용 절감 : 새로운 작업에 대해 매번 대량의 데이터를 수집하고 라벨링할 필요가 없어짐
 
- 평가 metrics
  - 일반적인 성능 평가 지표인 WER(Word Error Rate)
  - whisper의 경우 non-semantic difference는 제외하고 WER을 비교
 
# 5. 1년 사용 후기
## 장점
- 추가 학습 없이도 좋은 성능을 보여주는 오픈소스이며 준수한 성능
- 한국어의 경우 Common Voice 15, FLEURS 데이터셋에서 에러율이 낮은 언어 Top 3안에 속함
- sequence-level, word-level timestamp가 제공되어 자막에도 적합
- faster-whisper, whisper-jax, whisper-cpp, insanely-fast-whisper 등 다양한 구현체가 있어서 여러 use case에 따라 사용 가능

## 단점
- 학습 데이터 전처리 부족으로 인한 오류
  - whisper의 학습 데이터는 대부분 크롤링한 인터넷 데이터
  - 최근에 나온 단어 인식이 잘 되게 하는 장점 존재
    - ex) 비교적 최근에 등장한 단어인 MBTI, ITZY 등이 영어 대문자로 잘 전사
  - 하지만 데뷔 시기가 그 전인 뉴진스는 잘 인식 되지 않음
  - 음성이 없는데도 자막이 나타나는 경우 존재
    - 이는 음성 구간을 인식하는 VAD(Voice Activity Detection) filter를 같이 쓰면 어느정도 해결됨
- 여러 언어가 섞여 나오는 경우 언어 인식이 부정확함
  - whisper의 경우 30초씩 끊어서 추론하기 때문에 30초 내에 여러 언어가 나오면 한 언어로 도출되며 부정확한 결과 도출
  - 어떤 때는 우세한 언어로 번역되어 도출될때도 있고, 언어 기준 들리는대로 도출되는 등 일관되지 못한 결과 보임
- Transcribe와 Translate 강제가 잘 안 됨
  - whisper는 multi-task로 학습했기 때문에 Transcribe와 Translate 모드를 선택할 수 있음
  - 하지만 모드의 선택은 토큰을 시퀀스에 넣어서 처리하기 때문에 강제로 집행하기 어려움
  - 언어 고정을 하고 transcribe로 해도 자동으로 영어로 번역하는 등 제대로 작동되지 않음
- 추론 시간 계산의 어려움
  - ecoding 과정에서 temperature sampling 때문에 추론할 때마다 결과가 달라짐
- 발화가 멈췄다가 다시 시작했을 때 timestamp가 정확하지 않은 이슈
  - 긴 오디오를 추론할 때 발호가 멈췄다가 다시 나오는 경우 빈 구간 없이 앞부분의 발화와 뒷부분의 발화의 타임 스탬프가 합쳐져 길게 잡힐때 존재
  - 해당 이슈는 자막 표시에 큰 영향을 주기 때문에 어느 정도 튜닝을 해서 보정하는 것이 필요
- �가 나오는 이슈
  - 디코딩에 실패하는 경우 replacement character �(/ufffd)가 나올 수 있음
  - 이 부분은 regex로 영어,한국어,숫자,특수 문자 일부가 아닌 경우 다른 모델을 사용해서 추론하게 변경하거나 fine-tuning을 통해 해결가능
- 문장이나 단어가 반복되는 이슈
  - 다양한 시퀀스 to 시퀀스 모델에서 나타나는 이슈
  - 해당 부분은 compression_ratio_threshold 를 조절해서 반복을 줄여볼 수 있음
  - whisper에서는 디코딩 단계에서 repetition penalty를 주기 때문에 다른 모델에 비해 단어 반복이 적음

## 결론
- 보통 동시 발화가 없고 한 언어로 발화하는 오디오에 대해서는 위의 이슈가 나타나지 않음
- 계속해서 업데이트 되며 일부 이슈들은 해결이 되었을 수도 있음 (24.04)
