# Large Language Models are Efficient Learners of Noise-Robust Speech Recognition
## 요약 정리
### Problem


### Contributions


### Method


### Experiments



<br>  
  
## 0. Abstract
### 연구 문제 정의
- 기존 LLM 기반 GER은 깨끗한 음성에는 잘 동작하지만, 노이즈 환경에서의 강인성은 부족
- ASR에서 나온 N-best 후보 리스트가 noisy할 때 LLM으로 robust한 수정(correction)을 할 수 있을까?
- LLM에 “노이즈 상태”를 알려줘서 더 잘 고치게 할 방법 필요

### 기존 방법의 한계
- LM rescoring은 단순히 hypothesis의 점수 재조정 → 근본적인 수정은 어려움
- 기존 GER은 LLM의 강력한 언어 능력만으로 수정 → 노이즈 환경에서는 불안정
- 오디오 encoder의 audio embedding을 LLM에 바로 넣으면 cross-modal gap 때문에 tuning 성능 저하

### 제안 방법 : RobustGER
- **Language-space Noise Embedding (LSNE)**
  - N-best list의 diversity를 기반으로 언어적 noise condition 표현
  - Noise가 심할수록 → N-best diversity 커짐
  - Utterance-level diversity (전체 문장 의미 차이)
  - Token-level diversity (편집거리 중심 token 차이)
- **Audio Noise Distillation via MINE**
  - Mutual Information Neural Estimation(MINE) 활용
  - audio embedding의 noise 정보 → language embedding에 distill 
- **최종 Framework**
  - LLM fine-tuning 과정에서 LSNE를 noise conditioner로 투입
  - 목표: $Y = M_{H2T}(Y_N; -E_{LN})$

### 주요 실험 결과
- 최대 53.9% WER 감소
- Token-level embedding이 큰 효과 (WER metric에 직접 연관)
- Clean set에서도 30% WER 감소 → 일반화도 잘됨

### 논문의 장점 및 한계
- **장점**
  - LLM의 기존 능력을 Noise-robust ASR task로 효과적으로 확장
  - 모든 정보가 “언어적 표현” 안에서 처리 → cross-modal 문제 회피
  - 데이터 효율성 우수: 작은 training data로도 significant improvement
- **한계**
  - Context-based language correction과 noise-aware denoising trade-off 존재
  - CHiME-4에서 상대적으로 더 큰 개선 → dataset/domain bias 가능성 

