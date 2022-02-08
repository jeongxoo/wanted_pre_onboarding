# wanted_pre_onboarding
- 원티드 프리온보딩 AI/ML 선발 과제
- 과제 설명
https://codestates.notion.site/_AIB-8aaa720522d0496bb80a707f32dc7411

## Tokenizer() 풀이
**`1-1.preprocessing()`** -> 텍스트 전처리 함수
- input: 여러 영어 문장이 포함된 list
- output: 각 문장을 토큰화한 결과로, nested list
- 조건 1: 입력된 문장에 대해서 소문자로의 변환과 특수문자 제거를 수행합니다.
- 조건 2: 토큰화는 white space 단위로 수행합니다.


**`작성 코드`**
```
def preprocessing(self, sequences):
    fx_make = lambda x: re.sub(r"[^a-zA-Z0-9]", " ", x).lower().split()
    result = [fx_make(s) for s in sequences]
    return result
```

## 코드 작성 시 고려한 사항
특수 문자 제거 방법
- 정규식으로 특수 문자 제거
- str.isalnum으로 특수 문제 제거
    - 가독성 측면과 함수 실행 성능(속도) 측정 결과
    - 정규식을 채택하여 사용
- 형태소 분석기 사용
    - nltk 라이브러리 사용 고려  
    - but 직접 구현 가능 여부를 보여주는 것이 선발 과제에선 중요할 것으로 판단


---------
**`1-2.fit()`** -> 어휘 사전 구축 함수
- input: 여러 영어 문장이 포함된 list
- 조건 1: 위에서 만든 `preprocessing` 함수를 이용하여 각 문장에 대해 토큰화 수행
- 조건 2: 각각의 토큰을 정수 인덱싱 하기 위한 어휘 사전(`self.word_dict`) 생성
    - 주어진 코드에 있는 `self.word_dict` 활용


**`작성 코드`**
```
def fit(self, sequences):
    self.fit_checker = False

    # 조건 1 수행
    tokenized = self.preprocessing(sequences) 
    self.token = list(set(itertools.chain(*tokenized)))

    # 조건 2 수행
    _dict = {v: (i + 1) for i, v in enumerate(self.token)}
    self.word_dict = dict(_dict, **self.word_dict) 

    self.fit_checker = True
```
## 코드 작성 시 고려한 사항
preprocessing된 토큰 집합 생성 방법 (중복 X)
- reduce 사용
    - 극악의 성능으로 배제
- numpy(flatten, reshape 등) 사용
    - 기존 배열이 비정형 배열로 해당 기능 사용 불가 
- itertools 사용
    - itertools.chain.from_iterable()
    - itertools.chain()
        - 두 가지 모두 성능은 비슷
        - Pythonic한 Asterisk(*)을 사용하는 후자 채택

word_dict 생성 방법
- zip()
- dictionary comprehension
    - 두 방법 모두 성능은 유사
    - 조금 더 직관적이고, Pythonic한 dictionary comprehension 채택

word_dict, token 저장 방식
- 초기 작성 시에는 Tokenizer 클래스를 1회 호출하는 것을 가정하고 코드 작성
- 다만, TF-IDF 계산 시 매번 문장을 추가하여 계산하는 형태가 아니고
- batch 기반의 문장 데이터를 계산하는 것으로 판단하고 계산 마다 클래스를 호출하는 것으로 변경
- 따라서 word_dict, token에 대해서는 최대한 단순한 

---------
**`1-3.transform()`** -> 어휘 사전을 활용하여 입력 문장을 정수 인덱싱
- input: 여러 영어 문장이 포함된 list
- output: 각 문장의 정수 인덱싱으로, nested list
- 조건 1: 어휘 사전(`self.word_dict`)에 없는 단어는 'oov'의 index로 변환


**`작성 코드`**
```
def transform(self, sequences):
      result = []
      tokens = self.preprocessing(sequences)

      if self.fit_checker:
          fx_get_key = lambda x: x if x in self.word_dict.keys() else "oov"
          fx_get_val = lambda x: self.word_dict[fx_get_key(x)]
          result = [list(map(fx_get_val, t)) for t in tokens]

          return result
      else:
          raise Exception("Tokenizer instance is not fitted yet.")
```
## 코드 작성 시 고려한 사항
- 시간 복잡도를 고려하여 append() 함수 반복 대신
- list comprehension 활용
- 가독성 향상을 위해 파트별 lambda 함수 사용

---------

## TfidfVectorizer() 풀이
**`2-1.fit()`** -> 입력 문장들을 이용해 IDF 행렬을 만드는 함수
- input: 여러 영어 문장이 포함된 list
- 조건 1: IDF 행렬은 list 형태
    - ex) [토큰1에 대한 IDF 값, 토큰2에 대한 IDF 값, .... ]
- 조건 2: IDF 값은 아래 식을 이용해 구합니다.
    **`idf(d,t)=log_e({n}{1+df(d,t)})`**
    - **`df(d,t)`** : 단어 t가 포함된 문장 d의 개수
    - **`n`** : 입력된 전체 문장 개수
- 조건 3: 입력된 문장의 토큰화에는 문제 1에서 만든 Tokenizer 사용


**`작성 코드`**
```
def fit(self, sequences):
    tokenized = self.tokenizer.fit_transform(sequences)

    # list type의 df_matrix 생성
    fx_df = lambda x: sum([1 for tk in tokenized if x in tk])
    fx_get_idx = lambda x: self.tokenizer.word_dict[x]
    self.df_matrix = [fx_df(fx_get_idx(t)) for t in self.tokenizer.token]

    # idf 계산, 음수 처리 함수
    n = len(sequences)
    fx_idf = lambda x: math.log(n / (1 + x))
    fx_filter = lambda x: x if x >= 0 else 0

    # list type의 idf_matrix 생성
    self.idf_matrix = [fx_filter(fx_idf(df)) for df in self.df_matrix]  

    self.fit_checker = True
```

## 코드 작성 시 고려한 사항
idf_matrix 내 음수 제거
- 제공된 공식을 사용할 경우 모든 문장 내에 단어가 존재할 경우 idf 값은 음수가 됨
- idf 값이 음수인 경우, tf-idf 값도 음수가 됨
- 이 경우 음수는 의미 없는 값이기에 0으로 치환함


---------
**`2-2.transform()`** -> 입력 문장들을 이용해 TF-IDF 행렬을 만드는 함수
- input: 여러 영어 문장이 포함된 list
- output : nested list 형태입니다.
    ex) [[tf-idf(1, 1), tf-idf(1, 2), tf-idf(1, 3)], [tf-idf(2, 1), tf-idf(2, 2), tf-idf(2, 3)]]
- 조건1 : 입력 문장을 이용해 TF 행렬을 만드세요.
    - **`tf(d, t)`** : 문장 d에 단어 t가 나타난 횟수
- 조건2 : 문제 2-1( `fit()`)에서 만든 IDF 행렬과 아래 식을 이용해 TF-IDF 행렬을 만드세요
    - **`tf-idf(d,t) = tf(d,t) * idf(d,t)`**
        
        
**`작성 코드`**
```
def transform(self, sequences):
    if self.fit_checker:
        tokenized = self.tokenizer.transform(sequences)

        # idf 함수 -> [idf(1,1), idf(1,2) ... idf(1,n)]
        fx_get_idf = lambda y: np.array(list(map(lambda x: self.idf_matrix[x - 1], y)))

        # tf 함수 -> [tf(1,1), tf(1,2) ... tf(1,n)]
        fx_tf = lambda y: np.array(list(map(lambda x: y.count(x), y)))

        # array type의 tf, idf 곱셈 연산 후 list 형 변환 함수
        fx_multi = lambda x, y: (x * y).tolist()

        # tfidf 계산 함수, nested list 형태 출력
        self.tfidf_matrix = [fx_multi(fx_tf(tk), fx_get_idf(tk)) for tk in tokenized]

        return self.tfidf_matrix
    else:
        raise Exception("TfidfVectorizer instance is not fitted yet.")
```
## 코드 작성 시 고려한 사항
- 시간 복잡도를 고려하여 append() 함수 반복 대신
- list comprehension 활용
- tf * idf 계산 시에도 numpy array를 사용하여 반복문 없이 구현
- 다만, 과제에 주어진 nested list 타입을 반환하기 위해 형변환 과정이 한
- 가독성 향상을 위해 파트별 lambda 함수 사용


---------
