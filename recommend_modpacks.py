"""PyQt5 기반 Modrinth 모드팩 추천 앱.

추천 버튼을 눌렀을 때 실제 순서:
1. 검색어를 가져온다.
2. 체크박스가 켜져 있고 검색어가 영어가 아니면 번역기를 그때 처음 로드한다.
3. `preprocessor.py`와 같은 spaCy 전처리를 검색어에 적용한다.
4. Word2Vec으로 검색어 토큰을 확장한다.
5. 저장된 TF-IDF vectorizer로 검색어를 transform한다.
6. 저장된 TF-IDF matrix와 cosine similarity를 계산한다.
7. 드롭다운/체크박스 필터를 적용한다.
8. similarity와 downloads 기반 popularity를 섞어 최종 점수를 만든다.
9. 상위 결과를 테이블에 표시한다.
"""

from __future__ import annotations

import pickle
import sys
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from gensim.models import Word2Vec
from lingua import LanguageDetectorBuilder
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QHeaderView, QTableWidgetItem, QWidget
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import linear_kernel

from preprocessor import (
    clean_text,
    preprocess_english,
)
from translategemma_translator import (
    TranslateGemmaConfig,
    TranslateGemmaTranslator,
    load_translation_cache,
    save_translation_cache,
)


# =========================
# 파일 경로
# =========================

# Qt Designer 형식의 UI 파일. 버튼/드롭다운/체크박스/테이블 배치는 여기서 정의한다.
UI_PATH = Path(__file__).with_name("modpack_recommendation.ui")

# 전처리 완료 CSV. 모델 생성 때와 같은 방식으로 complete=1 row만 사용한다.
PREPROCESSED_CSV = Path("./datasets/modrinth_dataset_preprocessed.csv")

# `generate_model.py`가 만든 모델 산출물.
TFIDF_VECTORIZER_PATH = Path("./models/modpack_tfidf.pkl")
TFIDF_MATRIX_PATH = Path("./models/modpack_tfidf_matrix.npz")
WORD2VEC_MODEL_PATH = Path("./models/modpack_word2vec.model")


# =========================
# 추천 파라미터
# =========================

# 최종 화면에 표시할 추천 개수.
TOP_K = 100

# 먼저 TF-IDF 유사도 기준으로 이 개수만큼 후보를 좁힌 뒤 인기도를 섞는다.
# 후보 제한 없이 인기도를 섞으면 검색어와 덜 관련된 인기 모드팩이 올라올 수 있다.
CANDIDATE_SIZE = 2000

# 검색어 토큰 하나마다 Word2Vec 유사어를 몇 개까지 붙일지 정한다.
WORD2VEC_TOPN = 10

# 원본 검색어 토큰 반복 횟수.
# 참고 프로젝트에서 원본 keyword를 여러 번 반복해 가중치를 준 방식과 같은 의도다.
QUERY_TOKEN_WEIGHT = 11

# client_side/server_side 드롭다운 값.
SIDE_VALUES = ("any", "required", "optional", "unsupported", "unknown")


# =========================
# 검색어 번역/전처리 설정
# =========================

# 검색어 전처리에 사용할 spaCy 영어 모델. 문서 전처리와 같은 모델명을 기본값으로 둔다.
SPACY_MODEL_NAME = "en_core_web_sm"

# 검색어 번역은 전처리용 번역 설정과 분리한다. UI 검색만 다른 모델/양자화로 바꿀 때 여기만 수정한다.
USE_QUERY_TRANSLATION = True
QUERY_TRANSLATE_MODEL_ID = "google/translategemma-4b-it"
QUERY_TRANSLATE_MODEL_DIR = None
QUERY_TRANSLATION_CACHE = "./datasets/translation_cache.json"
QUERY_TRANSLATE_QUANTIZATION = "8bit"
QUERY_MODEL_DTYPE = "bfloat16"
QUERY_TRANSLATE_DEVICE_MAP = None

# 번역 입력/출력 토큰 예산. 입력이 길수록 출력 예산을 줄여 전체 컨텍스트 한도를 넘지 않게 한다.
QUERY_MAX_TRANSLATE_INPUT_TOKENS = 1024
QUERY_HARD_MAX_TRANSLATE_INPUT_TOKENS = 2000
QUERY_MAX_TRANSLATE_TOTAL_TOKENS = 2048
QUERY_MAX_TRANSLATE_OUTPUT_TOKENS = 2000
QUERY_MIN_TRANSLATE_OUTPUT_TOKENS = 1
QUERY_OUTPUT_TOKEN_RATIO = 1.6

# bitsandbytes 양자화 세부값. 4bit/8bit 설정을 바꿀 때 TranslateGemmaConfig로 그대로 전달된다.
QUERY_BNB_4BIT_QUANT_TYPE = "nf4"
QUERY_BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
QUERY_BNB_4BIT_USE_DOUBLE_QUANT = True
QUERY_BNB_8BIT_THRESHOLD = 6.0

# 번역 검수용 로그 설정. 언어 오인식이나 pad 출력 같은 문제를 확인하기 위해 검색어 번역도 로그를 남긴다.
QUERY_TRANSLATE_DEBUG_ON_FAILURE = True
QUERY_TRANSLATE_DEBUG_TO_CONSOLE = True
QUERY_TRANSLATE_DEBUG_DIR = "./logs/translation_debug"
QUERY_TRANSLATE_DEBUG_MAX_CHARS = 4000
QUERY_TRANSLATE_AUDIT_LOG = True
QUERY_TRANSLATE_AUDIT_DIR = "./logs/translation_audit"
QUERY_TRANSLATE_LOG_PROGRESS = True
QUERY_TRANSLATE_LOG_PREVIEW_CHARS = 120


# `.ui` 파일을 파이썬 클래스와 쓸 수 있게 로드한다.
FORM_CLASS = uic.loadUiType(str(UI_PATH))[0]

NUMERIC_SORT_ROLE = Qt.UserRole + 1


class SortableTableWidgetItem(QTableWidgetItem):
    """숫자 정렬값이 있으면 문자열 대신 숫자로 비교한다."""

    def __lt__(self, other: QTableWidgetItem) -> bool:
        left = self.data(NUMERIC_SORT_ROLE)
        right = other.data(NUMERIC_SORT_ROLE)

        if left is not None and right is not None:
            return float(left) < float(right)

        return super().__lt__(other)


class ModpackRecommendationApp(QWidget, FORM_CLASS):
    """추천 앱 메인 위젯.

    PyQt5에서는 버튼 클릭 같은 이벤트를 메서드에 연결해야 하므로 클래스는 필요하다.
    대신 추천 계산 자체는 `recommend_clicked()` 안에 선형적으로 배치했다.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setupUi(self)

        # =========================
        # 1. 전처리된 CSV 로드
        # =========================

        df = pd.read_csv(PREPROCESSED_CSV)

        # 전처리기는 실행 시작 시 실패 row를 제거한다. 추천기는 key 중복 제거를 하지 않고 CSV 순서를 그대로 쓴다.
        # 모델 생성도 같은 complete=1 필터만 적용하므로 DataFrame row와 TF-IDF matrix row가 같은 순서로 맞는다.
        complete_text = df["complete"].fillna("").astype(str).str.strip().str.lower()
        df = df[complete_text.isin({"1", "1.0", "true", "yes", "y"})].reset_index(drop=True)

        # description은 이미 전처리된 검색용 토큰 문자열이다. 빈 값 여부는 complete가 책임지므로 추가 필터링하지 않는다.
        df["description"] = df["description"].fillna("").astype(str)
        self.df = df

        # =========================
        # 2. 모델 산출물 로드
        # =========================

        # TF-IDF vectorizer는 검색어를 학습 때와 같은 feature 공간으로 바꿀 때 쓴다.
        with TFIDF_VECTORIZER_PATH.open("rb") as f:
            self.tfidf = pickle.load(f)

        # TF-IDF matrix는 각 모드팩 description의 벡터 행렬이다.
        self.tfidf_matrix = load_npz(TFIDF_MATRIX_PATH)

        # Word2Vec은 최종 추천 벡터가 아니라 검색어 확장에만 쓴다.
        self.word2vec = Word2Vec.load(str(WORD2VEC_MODEL_PATH))

        # row 수 검사는 중복/빈값을 보정하려는 코드가 아니라, CSV와 모델 산출물이 같은 시점인지 확인하는 안전장치다.
        # 전처리 CSV를 다시 만들었으면 TF-IDF matrix도 `generate_model.py`로 다시 만들어야 row index가 맞는다.
        if self.tfidf_matrix.shape[0] != len(self.df):
            raise RuntimeError(
                f"TF-IDF rows={self.tfidf_matrix.shape[0]}, CSV rows={len(self.df)}. "
                "Run generate_model.py again."
            )

        # =========================
        # 3. 검색어 전처리 도구 로드
        # =========================

        # spaCy는 검색어를 문서와 같은 영어 토큰 규칙으로 처리할 때 사용한다.
        self.nlp = spacy.load(SPACY_MODEL_NAME, disable=["parser", "ner"])

        # Lingua 언어 감지기는 비교적 작으므로 앱 시작 때 로드한다.
        # TranslateGemma 번역기는 매우 무거워서 여기서 로드하지 않고,
        # 실제 비영어 검색어가 들어왔을 때 `recommend_clicked()` 안에서 처음 만든다.
        self.language_detector = (
            LanguageDetectorBuilder
            .from_all_languages()
            .with_preloaded_language_models()
            .build()
        )
        self.translator = None
        self.translation_cache = None

        # =========================
        # 4. 드롭다운/체크박스/테이블 초기화
        # =========================

        self.cb_loader.addItem("any")
        loader_values = set()
        for cell in self.df["loaders"].fillna("").astype(str):
            # loaders 컬럼은 "fabric, quilt"처럼 쉼표 목록이므로 각 값을 드롭다운 후보로 모은다.
            for value in cell.split(","):
                value = value.strip().lower()
                if value:
                    loader_values.add(value)
        for value in sorted(loader_values):
            self.cb_loader.addItem(value)

        self.cb_game_version.addItem("any")
        version_values = set()
        for cell in self.df["game_versions"].fillna("").astype(str):
            # game_versions도 한 row에 여러 버전이 들어갈 수 있어 전체 CSV의 고유 버전을 수집한다.
            for value in cell.split(","):
                value = value.strip()
                if value:
                    version_values.add(value)
        for value in sorted(version_values):
            self.cb_game_version.addItem(value)

        self.cb_client_side.addItems(SIDE_VALUES)
        self.cb_server_side.addItems(SIDE_VALUES)

        # min downloads 체크박스를 꺼두면 spinbox 값은 무시한다.
        self.chk_min_downloads.setChecked(False)
        self.sb_min_downloads.setEnabled(False)

        # 결과 테이블은 이름과 URL을 넓게 보여주고, 나머지는 내용 크기에 맞춘다.
        self.tbl_results.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tbl_results.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tbl_results.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.tbl_results.horizontalHeader().setSectionResizeMode(7, QHeaderView.Stretch)

        # =========================
        # 5. PyQt signal 연결
        # =========================

        self.btn_recommend.clicked.connect(self.recommend_clicked)
        self.le_query.returnPressed.connect(self.recommend_clicked)
        self.sl_popularity.valueChanged.connect(self.update_popularity_label)
        self.chk_min_downloads.toggled.connect(self.sb_min_downloads.setEnabled)
        self.tbl_results.cellDoubleClicked.connect(self.open_result_url)

        self.lb_status.setText(f"Ready. modpacks={len(self.df)}")

    def update_popularity_label(self, value: int) -> None:
        """인기도 슬라이더 값이 바뀔 때 라벨을 갱신한다."""
        self.lb_popularity.setText(f"Popularity {value}")

    def recommend_clicked(self) -> None:
        """추천 버튼을 눌렀을 때 실행되는 전체 추천 절차."""

        # =========================
        # 1. 검색어 입력 확인
        # =========================

        raw_query = self.le_query.text().strip()
        if not raw_query:
            self.lb_status.setText("Enter a query.")
            return

        self.btn_recommend.setEnabled(False)
        self.lb_status.setText("Searching...")
        QApplication.processEvents()

        # =========================
        # 2. 검색어 정리 및 필요시 영어 번역
        # =========================

        # clean_text는 URL/Markdown/HTML/이모지 같은 검색 노이즈를 제거한다.
        english_query = clean_text(raw_query)
        detected_language = "disabled"

        if USE_QUERY_TRANSLATION and self.chk_translate.isChecked():
            language = self.language_detector.detect_language_of(english_query)
            detected_language = "unknown" if language is None else language.iso_code_639_1.name.lower()

            # 영어/unknown은 번역하지 않는다.
            # 비영어로 확실히 감지된 경우에만 무거운 TranslateGemma를 lazy load한다.
            if detected_language not in {"en", "unknown"}:
                if self.translation_cache is None:
                    self.translation_cache = load_translation_cache(QUERY_TRANSLATION_CACHE)

                if self.translator is None:
                    # 검색어 번역기를 이 시점에 처음 만든다. 앱 시작 때 만들면 첫 화면이 너무 늦어진다.
                    # 아래 값들은 이 파일 상단의 QUERY_* 설정이라 전처리기와 별개로 바꿀 수 있다.
                    self.translator = TranslateGemmaTranslator(TranslateGemmaConfig(
                        model_id=QUERY_TRANSLATE_MODEL_ID,
                        model_dir=QUERY_TRANSLATE_MODEL_DIR,
                        cache_path=QUERY_TRANSLATION_CACHE,
                        use_translation=USE_QUERY_TRANSLATION,
                        quantization=QUERY_TRANSLATE_QUANTIZATION,
                        model_dtype=QUERY_MODEL_DTYPE,
                        device_map=QUERY_TRANSLATE_DEVICE_MAP,
                        bnb_4bit_quant_type=QUERY_BNB_4BIT_QUANT_TYPE,
                        bnb_4bit_compute_dtype=QUERY_BNB_4BIT_COMPUTE_DTYPE,
                        bnb_4bit_use_double_quant=QUERY_BNB_4BIT_USE_DOUBLE_QUANT,
                        bnb_8bit_threshold=QUERY_BNB_8BIT_THRESHOLD,
                        max_input_tokens=QUERY_MAX_TRANSLATE_INPUT_TOKENS,
                        hard_max_input_tokens=QUERY_HARD_MAX_TRANSLATE_INPUT_TOKENS,
                        max_total_tokens=QUERY_MAX_TRANSLATE_TOTAL_TOKENS,
                        max_output_tokens=QUERY_MAX_TRANSLATE_OUTPUT_TOKENS,
                        min_output_tokens=QUERY_MIN_TRANSLATE_OUTPUT_TOKENS,
                        output_token_ratio=QUERY_OUTPUT_TOKEN_RATIO,
                        debug_on_failure=QUERY_TRANSLATE_DEBUG_ON_FAILURE,
                        debug_to_console=QUERY_TRANSLATE_DEBUG_TO_CONSOLE,
                        debug_dir=QUERY_TRANSLATE_DEBUG_DIR,
                        debug_max_chars=QUERY_TRANSLATE_DEBUG_MAX_CHARS,
                        audit_log=QUERY_TRANSLATE_AUDIT_LOG,
                        audit_dir=QUERY_TRANSLATE_AUDIT_DIR,
                        log_progress=QUERY_TRANSLATE_LOG_PROGRESS,
                        log_preview_chars=QUERY_TRANSLATE_LOG_PREVIEW_CHARS,
                    ))

                translated_query = self.translator.translate_text(
                    english_query,
                    detected_language,
                    "en",
                    self.translation_cache,
                    label="query",
                    raise_on_failure=False,
                )
                save_translation_cache(self.translation_cache, QUERY_TRANSLATION_CACHE)

                # 번역 실패 시 빈 문자열이 올 수 있다.
                # 검색 요청 전체를 죽이지 않고 원문을 그대로 전처리한다.
                if translated_query:
                    english_query = translated_query

        # =========================
        # 3. 검색어 전처리
        # =========================

        # 문서 description을 만들 때 쓴 것과 같은 preprocess_english를 검색어에도 적용한다.
        # 검색어에는 tags/categories가 없으므로 meta token 삽입은 하지 않는다.
        query_tokens = preprocess_english(english_query, self.nlp)

        if not query_tokens:
            self.tbl_results.setRowCount(0)
            self.lb_status.setText(f"No searchable tokens. language={detected_language}")
            self.btn_recommend.setEnabled(True)
            return

        # =========================
        # 4. Word2Vec 검색어 확장
        # =========================

        expanded_tokens = []
        for token in query_tokens:
            # 원본 검색어 토큰은 가장 중요한 신호이므로 많이 반복한다.
            expanded_tokens.extend([token] * QUERY_TOKEN_WEIGHT)

            # Word2Vec vocabulary에 없는 단어는 유사어만 못 붙일 뿐, 원본 토큰은 유지된다.
            if token not in self.word2vec.wv:
                continue

            # 가장 가까운 유사어부터 반복 횟수를 점점 줄여서 추가한다.
            similar_weight = QUERY_TOKEN_WEIGHT - 1
            for similar_word, _score in self.word2vec.wv.most_similar(token, topn=WORD2VEC_TOPN):
                expanded_tokens.extend([similar_word] * max(similar_weight, 1))
                similar_weight -= 1

        # 이론상 query_tokens가 있으면 expanded_tokens도 있어야 하지만, 안전하게 원본 토큰을 fallback으로 둔다.
        if not expanded_tokens:
            expanded_tokens = query_tokens

        # =========================
        # 5. TF-IDF transform 및 cosine similarity 계산
        # =========================

        # 검색어는 학습이 아니므로 fit_transform 금지.
        # 저장된 vectorizer의 transform만 사용해야 기존 matrix와 feature index가 같다.
        query_text = " ".join(expanded_tokens)
        query_vec = self.tfidf.transform([query_text])
        similarity = linear_kernel(query_vec, self.tfidf_matrix).ravel()

        # =========================
        # 6. 드롭다운/체크박스 필터 적용
        # =========================

        filter_mask = np.ones(len(self.df), dtype=bool)

        selected_loader = self.cb_loader.currentText().strip().lower()
        if selected_loader != "any":
            # 선택한 loader가 row의 loaders 목록 안에 정확히 있을 때만 통과시킨다.
            filter_mask &= self.df["loaders"].fillna("").astype(str).apply(
                lambda cell: selected_loader in {part.strip().lower() for part in cell.split(",") if part.strip()}
            ).to_numpy()

        selected_version = self.cb_game_version.currentText().strip()
        if selected_version != "any":
            # 선택한 버전도 쉼표 목록 안에서 정확히 일치하는 값만 통과시킨다.
            filter_mask &= self.df["game_versions"].fillna("").astype(str).apply(
                lambda cell: selected_version in {part.strip() for part in cell.split(",") if part.strip()}
            ).to_numpy()

        selected_client = self.cb_client_side.currentText().strip().lower()
        if selected_client != "any":
            # client_side는 단일 값 컬럼이라 문자열 비교만 하면 된다.
            filter_mask &= self.df["client_side"].fillna("unknown").astype(str).str.lower().eq(selected_client).to_numpy()

        selected_server = self.cb_server_side.currentText().strip().lower()
        if selected_server != "any":
            # server_side도 단일 값 컬럼이다.
            filter_mask &= self.df["server_side"].fillna("unknown").astype(str).str.lower().eq(selected_server).to_numpy()

        if self.chk_min_downloads.isChecked():
            # 최소 다운로드 체크박스가 켜진 경우에만 spinbox 값을 필터 조건으로 쓴다.
            downloads_for_filter = pd.to_numeric(self.df["downloads"], errors="coerce").fillna(0)
            filter_mask &= downloads_for_filter.ge(self.sb_min_downloads.value()).to_numpy()

        candidate_indices = np.flatnonzero(filter_mask)
        if candidate_indices.size == 0:
            self.tbl_results.setRowCount(0)
            self.lb_status.setText("No results after filters.")
            self.btn_recommend.setEnabled(True)
            return

        # 필터를 통과한 row 중 similarity가 높은 순서로 후보를 줄인다.
        candidate_indices = candidate_indices[np.argsort(similarity[candidate_indices])[::-1]]
        candidate_indices = candidate_indices[:CANDIDATE_SIZE]

        # =========================
        # 7. 인기도 점수와 최종 점수 계산
        # =========================

        downloads = pd.to_numeric(self.df["downloads"], errors="coerce").fillna(0).to_numpy(dtype=float)

        # 다운로드 수는 편차가 크므로 log1p로 눌러서 사용한다.
        popularity = np.log1p(downloads)
        if popularity.max(initial=0) > 0:
            popularity = popularity / popularity.max()

        # 슬라이더 0이면 similarity만, 100이면 popularity만 반영한다.
        popularity_weight = self.sl_popularity.value() / 100.0
        similarity_weight = 1.0 - popularity_weight
        final_score = similarity_weight * similarity[candidate_indices] + popularity_weight * popularity[candidate_indices]

        # final_score는 candidate_indices와 같은 순서이므로 정렬 결과를 candidate_indices에 다시 적용한다.
        final_order = np.argsort(final_score)[::-1][:TOP_K]
        ranked_indices = candidate_indices[final_order]

        # =========================
        # 8. 결과 테이블 표시
        # =========================

        # 정렬이 켜진 상태에서 row를 채우면 삽입 중 row 위치가 바뀔 수 있어 잠깐 끈다.
        self.tbl_results.setSortingEnabled(False)
        self.tbl_results.setRowCount(len(ranked_indices))

        for table_row, df_index in enumerate(ranked_indices):
            modpack = self.df.iloc[df_index]
            score_value = float(final_score[final_order][table_row])
            downloads_value = int(downloads[df_index])

            # 표시값은 문자열로 두되, 숫자 컬럼은 별도 정렬값을 함께 보관한다.
            values = [
                (f"{score_value:.4f}", score_value),
                (modpack.get("name", ""), None),
                (str(downloads_value), downloads_value),
                (modpack.get("loaders", ""), None),
                (modpack.get("game_versions", ""), None),
                (modpack.get("client_side", ""), None),
                (modpack.get("server_side", ""), None),
                (modpack.get("url", ""), None),
            ]

            for column, (display_value, sort_value) in enumerate(values):
                item = SortableTableWidgetItem(str(display_value))
                if sort_value is not None:
                    item.setData(NUMERIC_SORT_ROLE, sort_value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.tbl_results.setItem(table_row, column, item)

        self.tbl_results.setSortingEnabled(True)
        self.lb_status.setText(
            f"results={len(ranked_indices)} language={detected_language} tokens={' '.join(query_tokens)}"
        )
        self.btn_recommend.setEnabled(True)

    def open_result_url(self, row: int, _column: int) -> None:
        """결과 row를 더블클릭하면 마지막 URL 컬럼을 브라우저로 연다."""
        url_item = self.tbl_results.item(row, 7)
        if url_item and url_item.text().strip():
            webbrowser.open(url_item.text().strip())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModpackRecommendationApp()
    window.show()
    sys.exit(app.exec_())
