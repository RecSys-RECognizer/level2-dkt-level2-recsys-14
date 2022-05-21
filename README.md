# [Recsys-14] Deep Knowledge Tracing

<br/>

## 📚 프로젝트 개요

### 📋 프로젝트 주제

‘지식 상태’를 추적하는 딥러닝 방법론인 DKT(Deep Knowledge Tracing)를 통해 사용자가 풀어왔던 과목에 대해 얼만큼 이해하고 있는지를 측정하여 아직 풀지 않은 미래의 문제에 대해 맞을지 틀릴지 예측하는 모델 개발합니다.

<br/>

<br/>

### 📈 프로젝트 목표
주어진 문제는 학생 개개인의 이해도인 지식 상태를 예측뿐만 아니라 문제를 맞출 여부에 대한 확률로도 접근할 수 있습니다. 때문에 주어진 여러 추천 시스템 모델 뿐만 아니라 ML 계열의 모델과도 비교해보고, 이를 통해 Sequential data와 static data 각각에 관한 추천 시스템의 차이를 이해합니다. 또한 대회에서 주어지는 데이터 분석을 통해 최적의 모델과 추천 방식을 찾고, 이를 바탕으로 모델의 성능과 auroc, acc를 높이는 것을 목표로 합니다.

<br/>

<br/>

### 🗞 사용 데이터

**train_data.csv**

|  USER_ID  |  assessmentItemID  |    testID    | answerCode    | Timestamp    | KnowledgeTag    |
| :-------: | :-------: | :------------------: | :-------: | :-------: | :-------: |
| 사용자의 고유 번호 | 문항의 고유번호 | 사험지의 고유번호 |사용자가 해당 문항을 맞췄는지 여부(정답 1, 오답 0|사용자가 해당 문항을 풀기 시작한 시점|문항 당 하나씩 배정되는 태그(중분류)|

- 2,266,588개의 행으로 구성되었습니다.
- Implicit feedback 기반의 sequential recommendation 시나리오를 상정합니다.
- 사용자의 time-ordered sequence에서 일부 item이 누락(dropout)된 상태입니다.

<br/>

<br/>

### 🔑 프로젝트 특징

- DKT를 활용하면 학생 개개인에게 문제에 대한 이해도와 취약한 부분을 극복하기 위해 어떤 문제들을 풀면 좋을지 추천이 가능합니다.
- 학생 개개인의 이해도를 가리키는 지식 상태를 예측하는 일보다는, 주어진 문제를 맞출지 틀릴지 예측하는 것에 집중하여 해결할 수 있습니다.

<br/>

<br/>

### 🛎 요구사항

사용자의 문항의 고유번호, 시험지의 고유번호, 문제를 풀기 시작한 시점과 태그 등의 side-information을 활용하여 각 학생이 푼 문제 리스트와 정답 여부가 담긴 데이터를 받아 최종 문제를 맞출지 틀릴지 예측합니다.<br/>

<br/>

### 📌 평가 방법

DKT는 주어진 마지막 문제를 맞았는지 틀렸는지로 분류하는 이진 분류 문제이기에, AUROC(Area Under the ROC curve)와 Accuracy를 사용합니다

<br/>

<br/>

## 프로젝트 구조

### Flow Chart

![mermaid-diagram-20220417234600](https://cdn.jsdelivr.net/gh/Glanceyes/Image-Repository/2022/04/24/20220424_1650794858.png)

<br/>

<br/>

### 디렉토리 구조

```html
//input/data
├─ sample_submission.csv          # 📃 output sample 
├─ train_data.csv                 # 📁 train data set
└─ test_data.csv                  # 📁 test data set

/code
├── README.md
├── CatBoost
│   ├── config.py
│   ├── dataset.py
│   └── train_inference.py
├── LGBM
│   ├── args.py
│		├── asset
│		├── inference.py
│		├── lgbm
│		│   ├── model.py
│		│   ├── preprocess.py
│		│   └── trainer.py
│		├── model
│		├── output
│		├── save_pic
│		│   └── lgbm_feature_importance.png
│		└── train.py
├── LSTMAttn_with_LGCN
│   ├── criterion.py
│   ├── dataloader.py
│   ├── lightgcn
│   │   ├── config.py
│   │   ├── id2index.pickle
│   │   ├── inference.py
│   │   ├── install.sh
│   │   ├── lightgcn
│   │   │   ├── datasets.py
│   │   │   ├── models.py
│   │   │   ├── optimizer.py
│   │   │   ├── scheduler.py
│   │   │   └── utils.py
│   │   ├── train.py
│   │   └── weight
│   │       └── best_auc_model2.pt
│   ├── lightgcn_for_tag
│   │   ├── config.py
│   │   ├── inference.py
│   │   ├── install.sh
│   │   ├── lightgcn_for_tag
│   │   │   ├── datasets.py
│   │   │   ├── models.py
│   │   │   ├── optimizer.py
│   │   │   ├── scheduler.py
│   │   │   └── utils.py
│   │   ├── output
│   │   │   └── best_auc_submission.csv
│   │   ├── run.log
│   │   ├── tag2index.pickle
│   │   ├── temp.ipynb
│   │   ├── train.py
│   │   └── weight
│   │       └── best_auc_model_tag.pt
│   ├── metric.py
│   ├── model.py
│   ├── optimizer.py
│   ├── scheduler.py
│   ├── trainer.py
│   └── utils.py
├── README.md
├── args.py
├── hyper_run.py
├── id2index.pickle
├── inference.py
├── requirements.txt
├── tag2index.pickle
└── train.py
```

<br/>

<br/>

## 모델과 실험 내용
- 이번 프로젝트에서 실험한 단일 모델 종류는 Boosting 기법을 사용한 CatBoost, LightGBM, XGBoost 모델, RNN 계열의 Attention Mechanism을 사용한 LSTM 모델, GNN 계열의 LightGCN 모델이 있으며, 단일 모델 성능을 비교하여 최종 결과에 사용한 모델은 CatBoost, LightGBM, LSTM Attention(LSTM with attention mechanism)입니다.
    - 단일 모델 간 성능 실험 시 CatBoost, LightGBM, LSTM with attention mechanism 순으로 높았습니다.
        - Public AUC 기준 CatBoost 0.8139, LightGBM 0.7723, LSTM Attention 0.751
    - 기존 XGBoost(AUC 기준 0.6171)보다 LightGBM의 성능이 더 높았던 것은 LightGCN에서 사용하는 Gradient-based One Side Sampling 기법으로 결과적으로 상대적으로 작은 Gradient보다 큰 Gradient를 지니는 Instance에 초점을 두어 Underfitting을 막는 효과인 것으로 보입니다.
    - LightGCN의 그래프에서 학습한 유저와 태그에 대한 임베딩을 LSTM Attention의 Feature로 추가하여 최종적으로 CatBoost와 LightGBM을 앙상블 할 때 점수를 향상시킬 수 있었습니다.
        - Public AUC 기준 0.8168 → 0.8173
- CatBoost, LightGBM, LSTM Attention 세 모델을 앙상블하기 위해 Hard Voting을 진행했습니다.
    - 성능이 가장 좋았던 앙상블 모델 결과인 CatBoost에 가장 많은 가중치를 부여한 것이 Public AUC 기준으로 가장 좋은 성능을 보였습니다.
        - CatBoost : LightGBM : LSTM Attention = 3.5 : 1.5 : 1 (가중치)
    - LightGBM에서 예측한 결과를 가지고 OOF(Out Of Fold) Stacking 기법을 적용했을 때 Public AUC 기준 0.7881로 미적용한 실험보다 상대적으로 낮게 나왔지만, Private에서는 0.8369로 성능이 올랐습니다.
        - OOF Stacking 적용 시 메타 모델로는 XGBoost와 LightGBM을 사용했습니다.
        - OOF Stacking 기법에서 inference한 확률이 0 또는 1에 가깝게 극단적으로 나와서 Overfitting이 되었다고 예상해서 최종 제출 파일에 포함시키지 않았습니다.

<br/>

<br/>

## 💻 활용 도구 및 환경

- 코드 공유
    - GitHub
- 개발 환경
    - JupyterLab, VS Code, PyCharm
- 모델 성능 분석
    - Wandb

<br/>

<br/>

## 👩🏻‍💻👨🏻‍💻 팀원 소개

<table>
   <tr>
      <td align="center">진완혁</td>
      <td align="center">박정규</td>
      <td align="center">김은선</td>
      <td align="center">이선호</td>
      <td align="center">이서희</td>
   </tr>
   <tr height="160px">
       <td align="center">
         <a href="https://github.com/wh4044">
            <!--<img height="120px" weight="120px" src="/pictures/jwh.png"/>-->
            <img width="100" alt="스크린샷 2022-04-19 오후 5 44 23" src="https://user-images.githubusercontent.com/70509258/163962658-548a3022-bcd3-40c7-8ca1-88c7c417e1d9.png">
         </a>
      <td align="center">
         <a href="https://github.com/juk1329">
            <!--<img height="120px" weight="120px" src="https://avatars.githubusercontent.com/u/80198264?s=400&v=4"/>-->
            <img width="85" alt="스크린샷 2022-04-19 오후 5 47 38" src="https://user-images.githubusercontent.com/70509258/163963317-f074768e-8976-42c5-a595-3c5be8310f48.png">
         </a>
      </td>
      </td>
      <td align="center">
         <a href="https://github.com/sun1187">
            <!--<img height="120px" weight="120px" src="https://avatars.githubusercontent.com/u/70509258?v=4"/>-->
        <img width="104" alt="스크린샷 2022-04-19 오후 5 48 35" src="https://user-images.githubusercontent.com/70509258/163963764-b66c30fc-de18-46ff-a432-3cec6cd5f9a8.png">
 </a>
      </td>
      <td align="center">
         <a href="https://github.com/Glanceyes">
          <!--<img height="120px" weight="120px" src="https://cdn.jsdelivr.net/gh/Glanceyes/Image-Repository/2022/03/24/20220324_1648093619.jpeg"/>-->
            <img width="96" alt="스크린샷 2022-04-19 오후 5 49 21" src="https://user-images.githubusercontent.com/70509258/163964338-4fc7e32a-d00e-46f5-a514-7d07ff14bcbc.png">
 </a>
      </td>
      <td align="center">
         <a href="https://github.com/seo-h2">
            <!--<img height="120px" weight="120px" src="/pictures/seoh2.png"/>-->
            <img width="102" alt="스크린샷 2022-04-19 오후 5 49 30" src="https://user-images.githubusercontent.com/70509258/163964515-eb89af1f-d9af-4c67-8ea9-b283383d3199.png">
         </a>
      </td>
   </tr>
   <tr>
      <td align="center">Truth</td>
      <td align="center">Juke</td>
      <td align="center">Sunny</td>
      <td align="center">Glaneyes</td>
      <td align="center">Brill</td>
   </tr>
</table>

  <br/>

| 팀원   | 역할                                                         |
| ------ | ------------------------------------------------------------ |
| 김은선 | ML계열 기반 모델 실험과 feature engineering을 통한 LGBM 성능 향상, LightGCN validation set 변경과 튜닝을 통한 성능 향상 |
| 박정규 | EDA를 통해 후에 고민해볼 만한 feature 제시, CatBoost를 활용해 feature를 통한 Ensemble 실험, FE 실험 |
| 이서희 | EDA와 후시분석에 기반한 feature engineering / LSTM attention등 시퀀스 기반 모델 실험. LGCN으로 사전학습한 임베딩을 lstm에 반영하여 성능 개선 |
| 이선호 | 최근 사용자별 문제 풀이 이력, 태그 연속 출현 횟수 등 FE를 통해 CatBoost와 LightGBM 성능 향상. OOF Stacking 기법 실험과 앙상블 구현 |
| 진완혁 | 베이스라인 모델 성능 개선, ML계열 모델 실험을 통해 CatBoost 모델 적용. 하이퍼 파라미터 튜닝과 FE를 통해 단일 모델 성능 개선 |

<br/>
