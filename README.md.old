# news-topic-keyphrase-generation-model-dev


## Dev Objective

- 디코더 혹은 인코더-디코더 계열의 모델을 활용하여 신문기사를 입력으로 받아 토픽에 해당하는 key-phase를 생성하는 모델을 개발합니다.
- 모델 입출력 예시
  - 입력: NH농협은행, ‘농협이 곧 ESG’, 상생경영 시동건다 (...)
  - 출력: NH농협은행; ESG경영; 환경·사회책임·지배구조; 금융사; 비재무적 노력; 대한민국 리딩금융 ESG 어워드; 농협이 곧 ESG; 녹색금융 상품; NH친환경기업우대론; NH녹색성장론; 최우수상
<!-- - 한국어로 사전학습된 SBERT를 활용하여 KeyBERT 방법을 추가로 시험해볼 예정
  - [link to ref.](https://github.com/MaartenGr/KeyBERT) -->


## Model

- T5/GPT/BART 계열의 PLM 모델을 사용합니다.
- 주요 PLM: paust/pko-t5-base
<!-- - 본질적으로 요약 태스크에 준하는 경우이기 때문에 한국어 데이터를 활용해 신문기사 요약 학습이 되어 있는 모델을 추가로 탐색할 예정입니다. -->


## Data

- 웹크롤링을 통해 얻은 신문기사 제목과 본문(x)에 대하여 챗피티를 활용하여 토픽에 해당하는 key-phrase 라벨(y)을 생성하는 방식으로 구축된 데이터세트입니다.
- train/eval 데이터의 비율은 8:2로 합니다.
- V1 / V2
  - V1: chatgpt 프롬프트 엔지니어링으로 생성한 라벨을 작업자가 검수한 데이터 셋
  - V2: gpt4 프롬프트 엔지니어링으로 생성한 라벨 (v1에 비해 향상된 라벨 품질)
- V3
  - [link](#generate-more-samples-with-gpt-35-turbo)


## Metric

- Eval Loss를 기준으로 Best CKPT를 생성합니다.
  - 개발 모델이 생성하는 key-pharase가 훈련 데이터에 준하게 생성이 되는지 확인합니다.
- F1@K와 ROUGE 계산 함수를 작성하여 모델 개발에 활용합니다.
  - [link to F1@K ref.](https://huggingface.co/ml6team/keyphrase-generation-t5-small-inspec?text=In+this+work%2C+we+explore+how+to+learn+task+specific+language+models+aimed+towards+learning+rich+representation+of+keyphrases+from+text+documents.+We+experiment+with+different+masking+strategies+for+pre-training+transformer+language+models+%28LMs%29+in+discriminative+as+well+as+generative+settings.+In+the+discriminative+setting%2C+we+introduce+a+new+pre-training+objective+-+Keyphrase+Boundary+Infilling+with+Replacement+%28KBIR%29%2C+showing+large+gains+in+performance+%28up+to+9.26+points+in+F1%29+over+SOTA%2C+when+LM+pre-trained+using+KBIR+is+fine-tuned+for+the+task+of+keyphrase+extraction.+In+the+generative+setting%2C+we+introduce+a+new+pre-training+setup+for+BART+-+KeyBART%2C+that+reproduces+the+keyphrases+related+to+the+input+text+in+the+CatSeq+format%2C+instead+of+the+denoised+original+input.+This+also+led+to+gains+in+performance+%28up+to+4.33+points+inF1%40M%29+over+SOTA+for+keyphrase+generation.+Additionally%2C+we+also+fine-tune+the+pre-trained+language+models+on+named+entity+recognition%28NER%29%2C+question+answering+%28QA%29%2C+relation+extraction+%28RE%29%2C+abstractive+summarization+and+achieve+comparable+performance+with+that+of+the+SOTA%2C+showing+that+learning+rich+representation+of+keyphrases+is+indeed+beneficial+for+many+other+fundamental+NLP+tasks.)


## Progress

### Test Training

- 'paust/pko-t5-base'
  - [log](https://wandb.ai/dotsnangles/news-topic-keyphrase-generation-model-dev)
  - 코사인 스케쥴러를 적용하여 30에폭 훈련 진행 중이나 15에폭 이후로 로스와 루즈 스코어가 모두 정체
  - ROUGE1 기준으로 22를 넘기 어려울 것으로 보임
  - 훈련을 끝까지 마친 뒤 Best CKPT의 추론 결과를 토대로 훈련 데이터 보완 혹은 설계 개선이 필요할 수 있음
  - ainize/kobart-news 및 ainize/gpt-j-6B-float16를 훈련해 베이스라인을 잡을 예정
  - ainize/kobart-news의 경우 요약 데이터로 훈련이 된 모델이라 타 모델과 비교해 높은 성능을 낼 가능성이 있음
  - 현재 데이터의 샘플 별 키프레이즈의 갯수가 크게 상이하여 최대 5개에서 10개 사이로 제한해 전처리한 뒤 훈련을 진행해볼 필요가 있음
    - 토큰화 후 토큰 갯수가 2개부터 156개
    - keyphrase 객체수 분포 3분위수 기준 7개
    <!-- - ![num_of_keyphrasespng](images/num_of_keyphrasespng.jpg) -->
  - 보다 나은 품질의 v2 데이터에 키프레이즈 개체수를 7개로 제한하는 전처리 로직을 추가하여 훈련 예정
    - paust/pko-t5 / ainize/kobart-news / ainize/gpt-j-6B-float16 세 개 모델을 모두 테스트한 뒤 베이스라인을 잡음

#### retrospective after test training

- Data
  - news_topic_trainset2.json / news_topic_validset2.json
  - 추후 추가될 수도 있는 데이터와 현재 데이터의 일관성을 유지하기 위해 v2만 사용
    - GPT4로 라벨링한 데이터
  - 총 1585개 샘플
  - seperator는 ','
  - 추가적인 라벨 정제를 수행함
    - 앞뒤로 seperator가 붙어 있는 경우가 있어 삭제
    - 이상값에 가까울 정도로 키프레이즈가 많은 라벨이 있어 키프레이즈의 갯수를 7개로 제한
  - preprocess_v2.pickle
  - input data의 max_len을 좀 더 확보하기 위해 prefix를 간소화 ("generate keyphrases: ")

- Model
  - paust/pko-t5-base / ainize/kobart-news / ainize/gpt-j-6B-float16
  - ainize/gpt-j-6B-float16는 훈련에 필요한 리소스가 큰 관계로 작은 모델로 실험을 선 진행

- training args
  - batch_size: 2
  - learning_rate: 3e-6 * NGPU
  - lr_scheduler: cosine without warm-up
  - optimizer: adamw
  - metric_for_best_model: eval_loss
  - max_input_length: 512
  - max_target_length: 128
  - generation method for evaluation: greedy search

- Results
   - notebooks.old/papermill/.old 참조
   - F1 스코어를 메트릭으로 사용하는 것은 좀 더 고려가 필요할 것으로 보임
      - 키프레이즈를 완전히 동일하지는 않더라도 비슷하게 생성하는 경우나 생성된 키프레이즈의 순서가 라벨의 순서와 다를 경우 모두 오답 처리하는 방식이기 때문에 문제가 있어보임
   - 현재 ROUGE 스코어는 유니그램 기준으로 20 내외를 기록하고 있음
      - 이전에 진행했던 요약 태스크의 경우 70 초반에서 준수한 성능을 발휘했던 경험이 있으나 키프레이즈 생성의 경우 스코어 기준을 좀 더 낮게 잡아도 결과가 괜찮을 듯함
   - Greedy Search 방식으로 ROUGE1을 35 이상으로 올릴 수 있는 방법을 찾아봐야 함
      - Greedy Search 방식으로 어느 정도 정돈된 결과가 나온다면 빔서치나 샘플링 등의 제너레이션 기법을 적용해 결과를 확인해볼 수 있음
   - 영자신문 샘플이 몇몇 존재하는데 한국어 샘플에 대해서만 훈련을 진행하고 결과를 살펴볼 필요가 있음
   - 좀 더 실험을 진행해보고 결과가 미진할 경우 샘플을 1만건 정도 더 구축을 해보는 것도 방법일 듯함
      - 챗지피티 API를 통해 라벨링 자동화가 가능한지 확인해봐야 함 (가능)


### Generate more samples with GPT 3.5 Turbo

- [link](https://github.com/illunex/keyphrase-data-labelling-with-openai-api)
- openai api를 활용하여 1만건의 샘플을 추가로 생성합니다.
  - 프롬프트 엔지니어링 결과 예시
  - role: you are a data labeller who finds key-phrases in a news article.
  - response: Sure, I can help you find key-phrases in a news article. Please provide me with the article, and I will analyze it to identify the most important and relevant phrases.
  - prompt: find top 10 most important key-phrases in the article and separate the key-phrases with semi-colons; numbering is not needed; don't start or end it with any punctuation: NH농협은행, ‘농협이 곧 ESG’, 상생경영 시동건다 (...)
  - reponse: NH농협은행; ESG경영; 환경·사회책임·지배구조; 금융사; 비재무적 노력; 대한민국 리딩금융 ESG 어워드; 농협이 곧 ESG; 녹색금융 상품; NH친환경기업우대론; NH녹색성장론; 최우수상
- 한국어 신문기사만 사용
- 결과적으로 GPT 3.5 Turbo 수준의 key-phrase 생성 모델 구축을 목표로 함
<!-- - DB에 축적된 신문기사 샘플을 살펴볼 필요 -->

### paust/pko-t5-base with Data V3 (11683 samples)

- GPT 3.5 Turbo로 생성한 훈련 데이터 15000건 중 적합한 샘플 11683건을 훈련에 투입(train-eval ratio: 8:2)
- 시험 훈련과 비교를 위해 paust/pko-t5-base를 사용
- input_text의 토큰 길이는 500 이상 1000 미만으로 설정
  - 일반적인 신문기사의 길이를 고려
- output_text의 토큰 길이는 64 미만으로 설정
  - 과도하게 긴 구절이 포함된 라벨이 생성된 샘플을 제외
- notebooks.old/papermill/paust_pko_t5_base_v3_run_3.ipynb
- [log](https://wandb.ai/dotsnangles/news-topic-keyphrase-generation-model-dev?workspace=user-dotsnangles)

#### retrospective after run_3

- run_1과 run_2에서 개발 환경과 메트릭 함수의 작동상 오류가 있었음
- 수정 후 run_3 진행
- [result](notebooks.old/papermill/paust_pko_t5_base_v3_run_3.ipynb)
- 15에폭 훈련을 진행했으며 마지막에도 training loss가 eval loss보다 높아 좀 더 긴 훈련을 진행해볼 필요가 있음
- generation 시간이 오래 걸려 에폭마다 evaluation time이 긴 문제가 있었기 때문에 추후에는 eval loss를 모니터링하는 방식으로 훈련을 진행하고 best ckpt를 로드해 메트릭을 검증하는 쪽으로 가려고 함

#### retrospective after run_5

- Learning Rate와 Batch size를 환경에 맞춰 수정해 몇차례 시험 훈련을 추가로 진행
   - data v3 run_5 >>> 3e-6 / 24 / linear scheduler wo warm-up / 30 epochs
-  ROUGE 스코어 함수 수정 / F1@10 함수 정교화 / 자카드 유사도 함수 시험
   - F1@10함수를 유사한 키워드는 정답 처리하는 방식으로 수정
   - 'test_rouge1': 65.7355,
   - 'test_rouge2': 45.4681,
   - 'test_rougeL': 54.0561,
   - 'test_rougeLsum': 54.0561,
   - 'test_gen_len': 50.0941,
   - 'test_F1@10': 59.7773,
   - 'test_jaccard_similarity': 26.0938,
- Jaccard Similarity는 F1@K에 개념이 포함되어 있으며 유사한 키워드를 정답 처리하는 식으로 사용하기 어려움에 따라 참고용으로 사용
- 메트릭 함수를 조금 더 확실하게 검토한 뒤 훈련 데이터를 늘려 학습을 진행할 예정
- [추론 결과 링크 최하단 Generate 부분 참조](notebooks.old/papermill/paust_pko_t5_base_v3_run_5.ipynb)

### Todos after paust/pko-t5-base with Data V3

- 현재 정의된 전처리-훈련-추론-검증 파이프라인을 유지 [metrics](src/metrics.py)
   - ROUGE 및 F1@10을 검증에 사용하나 70~80점 달성시 구축 모델의 성능 측정은 휴먼 이밸류에이션을 기준으로 함
     - 현재 정의한 metrics는 GPT 3.5 Turbo가 생성한 라벨과 구축 모델의 추론 결과 간 유사성을 측정하는 척도로 참고
     - 70~80점 달성시 GPT 3.5 Turbo와 유사한 추론 결과가 생성되었다고 볼 수 있음
       - 현재 paust_pko_t5_base_v3_run_5의 경우 ROUGE@1과 F1@10이 각각 65.73와 59.77이나 실제 추론 결과를 살펴보면 이미 핵심어 추출의 품질이 준수하다고 판단됨 ([link 최하단 참조](notebooks.old/papermill/paust_pko_t5_base_v3_run_5.ipynb))
     - 본격적인 휴먼 이밸류에이션에서는 입력(x of eval dataset: title_content)과 추론 결과(y_hat: key-phrases)를 비교하여 구축 모델의 신문기사 내 핵심어 추출 성능을 판단
- 구축 모델이 다양한 신문기사에서 일관된 성능을 유지하기 위해서는 훈련 데이터 보완이 필요
  - 현재 확보한 훈련 샘플은 11683건
  - DB에서 불러온 30만건 중 라벨 생성을 위한 전처리를 거친 100886건의 샘플을 모두 라벨링하여 훈련 데이터로 사용
  - 훈련 가용 샘플 선택 과정에서의 손실율을 고려했을 때 총 7.8만건 정도의 훈련 샘플을 확보할 수 있을 것으로 예상
  - 현재 신문기사의 범주가 저장되어 있지 않아 신문기사 범주별 샘플 수를 파악할 수 없는 문제가 있음
    - 실제 모델을 사용하는 과정에서 특정 범주의 핵심어 추출 성능이 떨어진다면 추가 데이터를 확보해 후속 훈련을 진행하며 유지 및 보수할 예정
- 효율적인 훈련을 위해 LoRA(PEFT) 및 adafactor를 시험해볼 예정
  - 기본적으로는 현재까지 사용한 AdamW without LoRA를 유지

### paust/pko-t5-base & EleutherAI/polyglot-ko-1.3b with Data V4 (11683 samples) (ZeRO)

- 추가 라벨링을 실시한 dataset v4를 활용한 훈련
- train/eval sample ratio: 76472:3000
- ZeRO 기법 시험 및 적용
  - Stage 2/3 and CPU Off-load 시험 예정 (추후 LLM 운영 대비)
  - ZeRO의 경우 Adam/AdamW에 최적화되어 있기에 AdamW 사용
- paust/pko-t5-base와 EleutherAI/polyglot-ko-1.3b로 베이스라인 설정
- 정의한 metrics를 통해 사전 검증 후 휴먼 이밸류에이션 진행

#### run_1 (paust/pko-t5-base(0.22b))

- [log](https://wandb.ai/illunex_ai/news-topic-keyphrase-generation-model-dev)
- paust/pko-t5-base(0.22b)
- ZeRO 2 / batch_size 48 / num_train_epochs 50 / early_stopping_patience 3 (eval_loss)
- 'rouge1': 67.8227,
- 'rouge2': 47.6712,
- 'rougeL': 55.9405,
- 'rougeLsum': 55.9405,
- 'F1@10': 60.2782,
- 'jaccard_similarity': 27.6543,
- 'gen_len': 48.3867

#### LLM.int8() / LoRA / ZeRO Training Test (with LLaMA-7b / polyglot-ko-1.2b / polyglot-ko-5.8b)
- LLaMA-7b
  - GPT4ALL에 준비된 dataset와 training script로 훈련을 진행
  - context length 512로 1 batch per device 훈련이 가능했음
- polyglot-ko-5.8b
  - 현재 개발 중인 키프레이즈 추출 모델의 데이터세트의 context length의 범위는 대략 512에서 1024까지 (+64)
  - LLM.int8()과 LoRA를 활용한 Single GPU 훈련이 가능했음
- polyglot-ko-1.2b
  - LoRA / ZeRO 3 Offload를 활용해 18~24배치의 안정적인 훈련이 가능

#### run_2~7 (EleutherAI/polyglot-ko-1.3b)
- [log](https://wandb.ai/illunex_ai/news-topic-keyphrase-generation-model-dev)
- EleutherAI/polyglot-ko-1.3b
- LoRA / ZeRO 3 Offload / batch_size 18~24 / num_train_epochs 10 / early_stopping_patience 3 (eval_loss)
- learning rate test
  - 48e-6 per batch size 8
  - [polyglot-ko-1.3B pretrain info](https://wandb.ai/eleutherai/polyglot-ko/groups/polyglot-ko-1.3B)
- Decoder 모델 훈련 데이터 수정

#### run_8
- learning rate: 96e-6 per batch size 8
- [result](.log/eleutherai_polyglot_ko_1.3b_v4_run_8/eleutherai_polyglot_ko_1.3b_v4_run_8.txt)
- source와 target이 하나의 샘플로 묶여 CLM 방식으로 훈련이 진행되기 때문에 후처리 로직을 작성해야 할 필요가 있음
- 현재 source와 target의 prefix는 "generate keyphrases:" 및 "keyphrases generated:"
- 현재 생성 결과에 "keyphrases generated:" 이후 웹사이트 주소가 뜨는 경우가 있음

#### run_9~13
- source/target prefix를 토크나이저 사전에 추가한 뒤 훈련 (기존 학습 맥락에서 이탈된 토큰 사용 목적)
- run_13 result
  - 'rouge1': 50.1822,
  - 'rouge2': 35.8062,
  - 'rougeL': 42.1194,
  - 'rougeLsum': 42.1194,
  - 'F1@10': 61.0273,
  - 'jaccard_similarity': 25.3944,
  - 'gen_len': 64.0