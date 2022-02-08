# wanted_pre_onboarding
- 원티드 프리온보딩 AI/ML 선발 과제
- 과제 설명
https://codestates.notion.site/_AIB-8aaa720522d0496bb80a707f32dc7411

## Tokenizer() 풀이
**`preprocessing()`** -> 텍스트 전처리 함수
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

---------
**`fit()`** -> 어휘 사전 구축 함수
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
---------
**`transform()`** -> 어휘 사전을 활용하여 입력 문장을 정수 인덱싱
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
---------

## TfidfVectorizer() 풀이
**`fit()`** -> 입력 문장들을 이용해 IDF 행렬을 만드는 함수
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
---------
**`transform()`** -> 입력 문장들을 이용해 TF-IDF 행렬을 만드는 함수
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
---------
