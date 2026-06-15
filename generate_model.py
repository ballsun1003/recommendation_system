"""Modrinth 모드팩 추천용 TF-IDF / Word2Vec 모델 생성 스크립트.

이 파일은 참고 프로젝트의 `job03_TFIDF.py`, `job05_word2vec.py`처럼
"CSV를 읽고 -> 모델을 학습하고 -> models 폴더에 저장"하는 한 가지 일만 한다.

전제:
- `preprocessor.py`를 먼저 실행해 `datasets/modrinth_dataset_preprocessed.csv`를 만든다.
- 전처리기는 시작할 때 실패 row를 제거하므로 여기서는 CSV 행 순서를 그대로 믿는다.
- `description` 컬럼은 이미 공백 기준 검색 토큰 문자열이고, `complete=1` row만 학습한다.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from gensim.models import Word2Vec
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


# =========================
# 파일 경로
# =========================

# 전처리 완료 CSV. 추천 결과 표시/필터/인기도 계산에 필요한 원본 컬럼도 이 파일에 남아 있다.
PREPROCESSED_CSV = Path("./datasets/modrinth_dataset_preprocessed.csv")

# 모델 산출물 저장 폴더. `.gitignore`에 들어 있으므로 큰 모델 파일이 git에 올라가지 않는다.
MODELS_DIR = Path("./models")

# 추천 시 검색어를 같은 TF-IDF 공간으로 transform하기 위해 vectorizer 객체를 저장한다.
TFIDF_VECTORIZER_PATH = MODELS_DIR / "modpack_tfidf.pkl"

# 모든 모드팩 description의 TF-IDF sparse matrix.
TFIDF_MATRIX_PATH = MODELS_DIR / "modpack_tfidf_matrix.npz"

# 검색어 확장용 Word2Vec 모델.
WORD2VEC_MODEL_PATH = MODELS_DIR / "modpack_word2vec.model"


# =========================
# Word2Vec 학습 파라미터
# =========================

# 한 단어를 몇 차원 벡터로 표현할지 정한다. 너무 작으면 의미 표현이 부족하고,
# 너무 크면 데이터 규모에 비해 학습이 무거워진다.
WORD2VEC_VECTOR_SIZE = 1000

# 중심 단어 주변 몇 토큰까지 문맥으로 볼지 정한다. 값이 클수록 더 넓은 문맥을 본다.
WORD2VEC_WINDOW = 8

# 이 횟수보다 적게 나온 단어는 Word2Vec vocabulary에서 제외한다.
WORD2VEC_MIN_COUNT = 2

# Word2Vec 학습에 사용할 CPU worker 수.
WORD2VEC_WORKERS = 32

# 전체 말뭉치를 몇 번 반복 학습할지 정한다.
WORD2VEC_EPOCHS = 200

# 1이면 skip-gram, 0이면 CBOW.
# skip-gram은 희소한 도메인 단어의 주변 관계를 잡는 데 비교적 유리하다.
WORD2VEC_SG = 1


# =========================
# TF-IDF 학습 파라미터
# =========================

# 이 개수보다 적은 문서에 나온 토큰은 TF-IDF feature에서 제외한다.
TFIDF_MIN_DF = 2

# 전체 문서 중 이 비율보다 많이 등장한 토큰은 TF-IDF feature에서 제외한다.
TFIDF_MAX_DF = 0.90


# =========================
# 1. 전처리 CSV 로드
# =========================

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 전처리기가 실패 row를 지우고 성공 row를 append하므로, 모델 생성은 CSV 행 순서를 바꾸지 않는다.
df = pd.read_csv(PREPROCESSED_CSV)

# complete=1은 "전처리 성공"이라는 단일 기준이다. description 빈 값 검사 같은 2중 필터는 여기서 하지 않는다.
complete_text = df["complete"].fillna("").astype(str).str.strip().str.lower()
df = df[complete_text.isin({"1", "1.0", "true", "yes", "y"})].reset_index(drop=True)

# TF-IDF/Word2Vec 학습 텍스트는 description 하나뿐이다. name/downloads 같은 메타데이터는 추천 점수/표시에만 쓴다.
df["description"] = df["description"].fillna("").astype(str)

descriptions = df["description"]

# Word2Vec은 "문서 문자열"이 아니라 "토큰 리스트의 리스트"를 입력으로 받는다.
# description은 이미 전처리된 "token token token" 문자열이므로 split만 한다.
tokenized_descriptions = [text.split() for text in descriptions]


# =========================
# 2. TF-IDF vectorizer / matrix 생성
# =========================

tfidf = TfidfVectorizer(
    # 전처리기가 이미 토큰화를 끝냈으므로 정규식 토큰화를 하지 않고 공백 기준으로 나눈다.
    tokenizer=str.split,

    # 입력 문자열을 추가 가공하지 않는다. 소문자화/기호 제거는 전처리 단계에서 끝났다.
    preprocessor=None,

    # tokenizer를 직접 지정할 때 sklearn 기본 token_pattern과 충돌하지 않게 끈다.
    token_pattern=None,

    # preprocessor.py가 이미 소문자로 정규화했으므로 여기서 다시 lower하지 않는다.
    lowercase=False,

    # 너무 희소한 토큰 제거 기준.
    min_df=TFIDF_MIN_DF,

    # 너무 흔한 토큰 제거 기준.
    max_df=TFIDF_MAX_DF,

    # 단어 빈도 tf를 그대로 쓰지 않고 1 + log(tf)로 눌러서 반복 단어의 과한 영향력을 줄인다.
    sublinear_tf=True,

    # 각 문서 벡터를 L2 정규화한다. 이렇게 하면 cosine similarity 계산이 안정적이다.
    norm="l2",
)

# fit_transform은 학습 시점에만 사용한다.
# 추천 시점에는 저장된 vectorizer로 검색어를 transform만 해야 feature index가 맞는다.
tfidf_matrix = tfidf.fit_transform(descriptions)

with TFIDF_VECTORIZER_PATH.open("wb") as f:
    pickle.dump(tfidf, f)
save_npz(TFIDF_MATRIX_PATH, tfidf_matrix)


# =========================
# 3. Word2Vec 모델 생성
# =========================

word2vec = Word2Vec(
    sentences=tokenized_descriptions,
    vector_size=WORD2VEC_VECTOR_SIZE,
    window=WORD2VEC_WINDOW,
    min_count=WORD2VEC_MIN_COUNT,
    workers=WORD2VEC_WORKERS,
    epochs=WORD2VEC_EPOCHS,
    sg=WORD2VEC_SG,
)
word2vec.save(str(WORD2VEC_MODEL_PATH))


# =========================
# 4. 생성 결과 확인용 출력
# =========================

print(f"rows={len(df)}")
print(f"tfidf_shape={tfidf_matrix.shape}")
print(f"tfidf_vocab={len(tfidf.vocabulary_)}")
print(f"word2vec_vocab={len(word2vec.wv.index_to_key)}")
print(f"saved={TFIDF_VECTORIZER_PATH}")
print(f"saved={TFIDF_MATRIX_PATH}")
print(f"saved={WORD2VEC_MODEL_PATH}")
