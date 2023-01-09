# KoBERT-based-simple-korean-semantic-analyzer

KoBERT을 이용하여 [AIHub][https://www.aihub.or.kr/aihubdata/data/view.do?currmenu=115&topmenu=100&aihubdatase=realm&datasetsn=86]의 데이터를 이용하여 구축한 감정분석기입니다.

현재 Test.py에는 테스트를 위한 간단한 문장을 작성하였습니다.

<img width="1273" alt="image" src="https://user-images.githubusercontent.com/35767154/211251775-a1eb958c-a7cc-4863-8c14-4b77148fef42.png">

# 사용법

docker로 구축하였으며, arm64에서 구동되게 끔 컨테이너를 작성했습니다

> ghcr.io/aglide100/kobert-based-simple-korean-semantic-analyzer

> docker run ghcr.io/aglide100/kobert-based-simple-korean-semantic-analyzer Test.py
