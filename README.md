# Data analysis case study

## FYI
- takes too much time for Support Vector Machine as classifier,
  - instead SVM, recommend random forest
- StandardScaler(), RobustScaler() -> is there much difference?
- Dask for big data, but failed to use 

## Dataset
- Regine comics data challenge 2017
<https://tech.lezhin.com/events/data-challenge-pyconkr-2017>
## Binary classification
* Purchase prediction model
* Result evaluation
    * Accuracy
    * ROC curve

## Data description
* file : tsv <br>
* 파일 포맷: TSV<br>
* 파일 용량: 228M (압축해서 26M)<br>
* 샘플 수:<br>
    * training : 650,965 <br>
    * train:
* number of feature: 167 개<br>
1 : label 해당 유저가 목록에 진입하고 1시간 이내에 구매했는지 여부<br>
2 : 사용 플랫폼 A<br>
3 : 사용 플랫폼 B<br>
4 : 사용 플랫폼 C<br>
5 : 사용 플랫폼 D<br>
6 : 목록 진입시점 방문 총 세션 수 (범위별로 부여된 순차 ID)<br>
7 : 작품을 나타내는 해쉬<br>
8-10 : 개인정보<br>
11-110 : 주요 작품 구매 여부<br>
111 : 작품 태그 정보<br>
112 : 구매할 때 필요한 코인<br>
113 : 완결 여부<br>
114-123 : 스케쥴 정보<br>
124-141 : 장르 정보<br>
142 : 해당 작품의 마지막 에피소드 발행 시점 (범위별로 부여된 순차 ID)<br>
143 : 단행본 여부<br>
144 : 작품 발행 시점 (범위별로 부여된 순차 ID)<br>
145 : 총 발행 에피소드 수 (범위별로 부여된 순차 ID)<br>
146-151 : 작품 태그 정보<br>
152-167 : 유저의 성향 정보 (과거에 구매를 했을 때만 기록)<br>

