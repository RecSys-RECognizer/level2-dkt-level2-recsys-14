# 모델 개요
- lightgcn으로부터 학습된 아이템 임베딩과 태그 임베딩을 lstm attention에 반영하여 학습하는 코드입니다.
<br/>

# 실행 방법
1. python ./LSTMAttn_with_LGCN/lightgcn/train.py *아이템 임베딩 추출*
2. python ./LSTMAttn_with_LGCN/lightgcn_for_tag/train.py *태그 임베딩 추출*
3. python train.py *임베딩 반영한 lstm attention 학습*