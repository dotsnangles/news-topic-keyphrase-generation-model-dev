# news-topic-keyphrase-generation-model-dev


## Dev Objective

- 디코더 혹은 인코더-디코더 계열의 모델을 활용하여 신문기사를 입력으로 받아 토픽에 해당하는 key-phase를 생성하는 모델을 개발합니다.
- 모델 입출력 예시
  - 입력: 다음 달부터 15억 초과 주택 대출 허용…ltv 50%도 다음 달부터는 무주택자의 경우 규제 지역에서도 주택담보대출비율(ltv)이 (...)
  - 출력: 주택담보대출,ltv 규제 완화,dsr
- 한국어로 사전학습된 SBERT를 활용하여 KeyBERT 방법을 추가로 시험해볼 예정
  - [link to ref.](https://github.com/MaartenGr/KeyBERT)


## Model

- T5/GPT/BART 계열의 모델을 사용합니다.
- 본질적으로 요약 태스크에 준하는 경우이기 때문에 한국어 데이터를 활용해 신문기사 요약 학습이 되어 있는 모델을 추가로 탐색할 예정입니다.


## Data

- 웹크롤링을 통해 얻은 신문기사 제목과 본문(x)에 대하여 챗피티를 활용하여 토픽에 해당하는 key-phrase 라벨(y)을 생성하는 방식으로 구축된 데이터세트입니다.
- 현재 v1과 v2가 존재하며 시험 훈련에서는 2513개의 전체 샘플을 활용합니다.
  - v1: chatgpt 프롬프트 엔지니어링으로 생성한 라벨을 작업자가 검수한 데이터 셋
  - v2: gpt4 프롬프트 엔지니어링으로 생성한 라벨 (v1에 비해 향상된 라벨 품질)
- train/eval 데이터는 초기 훈련에서 전체 데이터를 8:2로 나눠서 사용합니다.


## Metric

- 시험 훈련에서는 Eval Loss를 기준으로 오버피팅 전에 얼리 스타핑하여 휴먼 이밸류에이션을 진행합니다.
  - 개발 모델이 생성하는 key-pharase가 훈련 데이터에 준하게 생성이 되는지 확인합니다.\
- 추후에는 F1 스코어 계산 함수를 작성하여 모델 개발에 활용합니다.'
  - [link to ref.](https://huggingface.co/ml6team/keyphrase-generation-t5-small-inspec?text=In+this+work%2C+we+explore+how+to+learn+task+specific+language+models+aimed+towards+learning+rich+representation+of+keyphrases+from+text+documents.+We+experiment+with+different+masking+strategies+for+pre-training+transformer+language+models+%28LMs%29+in+discriminative+as+well+as+generative+settings.+In+the+discriminative+setting%2C+we+introduce+a+new+pre-training+objective+-+Keyphrase+Boundary+Infilling+with+Replacement+%28KBIR%29%2C+showing+large+gains+in+performance+%28up+to+9.26+points+in+F1%29+over+SOTA%2C+when+LM+pre-trained+using+KBIR+is+fine-tuned+for+the+task+of+keyphrase+extraction.+In+the+generative+setting%2C+we+introduce+a+new+pre-training+setup+for+BART+-+KeyBART%2C+that+reproduces+the+keyphrases+related+to+the+input+text+in+the+CatSeq+format%2C+instead+of+the+denoised+original+input.+This+also+led+to+gains+in+performance+%28up+to+4.33+points+inF1%40M%29+over+SOTA+for+keyphrase+generation.+Additionally%2C+we+also+fine-tune+the+pre-trained+language+models+on+named+entity+recognition%28NER%29%2C+question+answering+%28QA%29%2C+relation+extraction+%28RE%29%2C+abstractive+summarization+and+achieve+comparable+performance+with+that+of+the+SOTA%2C+showing+that+learning+rich+representation+of+keyphrases+is+indeed+beneficial+for+many+other+fundamental+NLP+tasks.)


## Progress

1. 'paust/pko-t5-base'
   - [log](https://wandb.ai/dotsnangles/news-topic-keyphrase-generation-model-dev)
   - 코사인 스케쥴러를 적용하여 30에폭 훈련 진행 중이나 15에폭 이후로 로스와 루즈 스코어가 모두 정체
   - ROUGE1 기준으로 22를 넘기 어려울 것으로 보임
   - 훈련을 끝까지 마친 뒤 Best CKPT의 추론 결과를 토대로 훈련 데이터 보완 혹은 설계 개선이 필요할 수 있음
   - ainize/kobart-news 및 ainize/gpt-j-6B-float16를 훈련해 베이스라인을 잡을 예정
   - ainize/kobart-news의 경우 요약 데이터로 훈련이 된 모델이라 타 모델과 비교해 높은 성능을 낼 가능성이 있음