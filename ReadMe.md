# KoBERT-based-simple-korean-semantic-analyzer

pretrained 된 [SKT KoBERT](https://github.com/SKTBrain/KoBERT)과 [AIHub](https://www.aihub.or.kr/aihubdata/data/view.do?currmenu=115&topmenu=100&aihubdatase=realm&datasetsn=86)의 감정 말뭉치 데이터를 이용하여 구축한 감정분석기입니다.

- 기쁨
- 불안
- 당황
- 슬픔
- 분노
- 상처

를 도메인으로 하여, KoBERT에서 제공되는 모델 학습 가이드에서 그렇게 큰 차이 없이 학습하여 사용하였습니다.

현재 Test.py에는 테스트를 위한 간단한 문장을 작성하였습니다.

<img width="1273" alt="image" src="https://user-images.githubusercontent.com/35767154/211251775-a1eb958c-a7cc-4863-8c14-4b77148fef42.png">

# 사용법

docker로 구축하였으며, pytorch를 구동하기 위해 필요한 의존성을 가져와서 사용하고 있습니다. 

또한 arm64에서 구동되게 끔 컨테이너를 작성했습니다. 

다만 docker 컨테이너의 특성상 gpu을 통한 추론은 힘들것으로 생각됩니다. 

> docker pull ghcr.io/aglide100/kobert-based-simple-korean-semantic-analyzer

> docker run ghcr.io/aglide100/kobert-based-simple-korean-semantic-analyzer Test.py
