"""Modrinth modpack CSV preprocessor.

이 파일은 크롤러가 만든 원본 CSV를 추천 모델 학습용 CSV로 바꾼다.
큰 흐름은 참고 프로젝트처럼 단순하게 유지한다.

1. 원본 CSV 로드
2. description + body 정리
3. Lingua로 언어 판정 후 필요한 경우에만 영어 번역
4. spaCy로 영어 검색용 토큰 생성
5. tags/categories 메타 토큰 삽입
6. 최종 CSV에 append하고 complete 상태 기록
"""

from __future__ import annotations

import html
import re
import time
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import spacy
from lingua import LanguageDetectorBuilder
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SKLEARN_STOP_WORDS

from translategemma_translator import (
    TranslateGemmaConfig,
    TranslateGemmaTranslator,
    load_translation_cache,
    save_translation_cache,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        # tqdm이 설치되지 않아도 전처리 자체는 진행되게 원본 iterable을 그대로 돌려준다.
        return iterable


# =========================
# 경로와 실행 설정
# =========================

INPUT_CSV = "./datasets/modrinth_dataset.csv"
OUTPUT_CSV = "./datasets/modrinth_dataset_preprocessed.csv"
TRANSLATION_CACHE = "./datasets/translation_cache.json"

# 영어 토큰화, 품사 태깅, 원형화에 사용할 spaCy 파이프라인이다.
SPACY_MODEL_NAME = "en_core_web_sm"

USE_TRANSLATION = True
TRANSLATE_MODEL_ID = "google/translategemma-12b-it"
TRANSLATE_MODEL_DIR = None
TRANSLATE_QUANTIZATION = "8bit"
MODEL_DTYPE = "bfloat16"
TRANSLATE_DEVICE_MAP = None

# 4bit/8bit 양자화 세부 설정은 TranslateGemmaConfig로 그대로 전달된다.
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
BNB_4BIT_USE_DOUBLE_QUANT = True
BNB_8BIT_THRESHOLD = 6.0

# 입력 chunk와 생성 출력이 모델 전체 토큰 예산 안에 들어가도록 번역기가 사용하는 값이다.
MAX_TRANSLATE_INPUT_TOKENS = 1024
HARD_MAX_TRANSLATE_INPUT_TOKENS = 2000
MAX_TRANSLATE_TOTAL_TOKENS = 2048
MAX_TRANSLATE_OUTPUT_TOKENS = 2000
MIN_TRANSLATE_OUTPUT_TOKENS = 1
OUTPUT_TOKEN_RATIO = 1.6

# 실패 디버그와 번역 감사 로그는 언어 오인식/잘림/pad 출력 검수에 사용한다.
TRANSLATE_DEBUG_ON_FAILURE = True
TRANSLATE_DEBUG_TO_CONSOLE = True
TRANSLATE_DEBUG_DIR = "./logs/translation_debug"
TRANSLATE_DEBUG_MAX_CHARS = 4000
TRANSLATE_AUDIT_LOG = True
TRANSLATE_AUDIT_DIR = "./logs/translation_audit"
TRANSLATE_LOG_PROGRESS = True
TRANSLATE_LOG_PREVIEW_CHARS = 120

# META_INTERVAL은 본문 토큰 사이에 tags/categories 메타 토큰을 다시 넣는 간격이다.
META_INTERVAL = 30
CACHE_SAVE_EVERY = 20
PRINT_ROW_TIMING = True


# =========================
# 영어 전처리 설정
# =========================

TOKEN_RE = re.compile(r"[A-Za-z0-9+#][A-Za-z0-9+#._-]*")
MINECRAFT_VERSION_RE = re.compile(r"^1\.\d+(?:\.\d+)?$")
PACK_VERSION_RE = re.compile(r"\bv\d+(?:\.(?:\d+|x)){0,3}(?:\s*[~\-–]\s*v?\d+(?:\.(?:\d+|x)){0,3})*\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"^\d+(?:\.\d+)*$")
EMOJI_RE = re.compile(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+")
KEEP_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
PHRASE_SEP = r"[\s._/'’-]+"

# 본문마다 띄어쓰기/하이픈/별칭이 다르게 적히는 다단어 표현을 하나의 검색 토큰으로 맞춘다.
DOMAIN_PHRASES = (
    (re.compile(rf"\bquality{PHRASE_SEP}of{PHRASE_SEP}life\b", re.IGNORECASE), "qol"),
    (re.compile(rf"\bclient{PHRASE_SEP}side\b", re.IGNORECASE), "clientside"),
    (re.compile(rf"\bserver{PHRASE_SEP}side\b", re.IGNORECASE), "serverside"),
    (re.compile(rf"\bsingle{PHRASE_SEP}players?\b", re.IGNORECASE), "singleplayer"),
    (re.compile(rf"\bmulti{PHRASE_SEP}players?\b", re.IGNORECASE), "multiplayer"),
    (re.compile(rf"\bresource{PHRASE_SEP}packs?\b", re.IGNORECASE), "resourcepack"),
    (re.compile(rf"\btexture{PHRASE_SEP}packs?\b", re.IGNORECASE), "texturepack"),
    (re.compile(rf"\bshader{PHRASE_SEP}packs?\b", re.IGNORECASE), "shaderpack"),
    (re.compile(rf"\bdata{PHRASE_SEP}packs?\b", re.IGNORECASE), "datapack"),
    (re.compile(rf"\bmod{PHRASE_SEP}loaders?\b", re.IGNORECASE), "modloader"),
    (re.compile(rf"\bworld{PHRASE_SEP}gen(?:eration)?\b", re.IGNORECASE), "worldgen"),
    (re.compile(rf"\bmini{PHRASE_SEP}map\b", re.IGNORECASE), "minimap"),
    (re.compile(rf"\bsky{PHRASE_SEP}block\b", re.IGNORECASE), "skyblock"),
    (re.compile(rf"\bone{PHRASE_SEP}block\b", re.IGNORECASE), "oneblock"),
    (re.compile(rf"\bstone{PHRASE_SEP}block\b", re.IGNORECASE), "stoneblock"),
    (re.compile(rf"\bkitchen{PHRASE_SEP}sink\b", re.IGNORECASE), "kitchensink"),
     (re.compile(rf"\bvanilla{PHRASE_SEP}plus\b", re.IGNORECASE), "vanillaplus"),
    (re.compile(r"\bvanilla\s*\+(?=\W|$)", re.IGNORECASE), "vanillaplus"),
    (re.compile(rf"\blow(?:er)?{PHRASE_SEP}end\b", re.IGNORECASE), "lowend"),
    (re.compile(rf"\bhigh{PHRASE_SEP}end\b", re.IGNORECASE), "highend"),
    (re.compile(rf"\bearly{PHRASE_SEP}game\b", re.IGNORECASE), "earlygame"),
    (re.compile(rf"\blate{PHRASE_SEP}game\b", re.IGNORECASE), "lategame"),
    (re.compile(rf"\bend{PHRASE_SEP}game\b", re.IGNORECASE), "endgame"),
    (re.compile(rf"\bopen{PHRASE_SEP}world\b", re.IGNORECASE), "openworld"),
    (re.compile(rf"\bboss{PHRASE_SEP}fights?\b", re.IGNORECASE), "bossfight"),
    (re.compile(rf"\bboss{PHRASE_SEP}battles?\b", re.IGNORECASE), "bossbattle"),
    (re.compile(rf"\bquest{PHRASE_SEP}book\b", re.IGNORECASE), "questbook"),
    (re.compile(rf"\bskill{PHRASE_SEP}tree\b", re.IGNORECASE), "skilltree"),
    (re.compile(rf"\btech{PHRASE_SEP}tree\b", re.IGNORECASE), "techtree"),
    (re.compile(rf"\bsimple{PHRASE_SEP}voice{PHRASE_SEP}chat\b", re.IGNORECASE), "simplevoicechat"),
    (re.compile(rf"\bvoice{PHRASE_SEP}chat\b", re.IGNORECASE), "voicechat"),
    (re.compile(rf"\bproximity{PHRASE_SEP}chat\b", re.IGNORECASE), "proximitychat"),
    (re.compile(rf"\bchat{PHRASE_SEP}reports?\b", re.IGNORECASE), "chatreports"),
    (re.compile(rf"\bdynamic{PHRASE_SEP}fps\b", re.IGNORECASE), "dynamicfps"),
    (re.compile(rf"\bforge{PHRASE_SEP}config{PHRASE_SEP}api(?:{PHRASE_SEP}port)?\b", re.IGNORECASE), "api"),
    (re.compile(rf"\bapplied{PHRASE_SEP}energistics(?:{PHRASE_SEP}2)?\b", re.IGNORECASE), "ae2"),
    (re.compile(rf"\brefined{PHRASE_SEP}storage\b", re.IGNORECASE), "refinedstorage"),
    (re.compile(rf"\bthermal{PHRASE_SEP}expansion\b", re.IGNORECASE), "thermalexpansion"),
    (re.compile(rf"\bfarmer(?:['’]s|s)?{PHRASE_SEP}delight\b", re.IGNORECASE), "farmersdelight"),
    (re.compile(rf"\bdistant{PHRASE_SEP}horizons\b", re.IGNORECASE), "distanthorizons"),
    (re.compile(rf"\bxaero(?:['’]s|s)?{PHRASE_SEP}world{PHRASE_SEP}map\b", re.IGNORECASE), "xaerosworldmap"),
    (re.compile(rf"\bxaero(?:['’]s|s)?{PHRASE_SEP}minimap\b", re.IGNORECASE), "xaerosminimap"),
    (re.compile(rf"\bworld{PHRASE_SEP}map\b", re.IGNORECASE), "worldmap"),
    (re.compile(rf"\btom(?:['’]s|s)?{PHRASE_SEP}simple{PHRASE_SEP}storage\b", re.IGNORECASE), "tomssimplestorage"),
    (re.compile(rf"\bentity{PHRASE_SEP}texture{PHRASE_SEP}features\b", re.IGNORECASE), "entitytexturefeatures"),
    (re.compile(rf"\bentity{PHRASE_SEP}culling\b", re.IGNORECASE), "entityculling"),
    (re.compile(rf"\bcreate{PHRASE_SEP}steam{PHRASE_SEP}(?:n|and){PHRASE_SEP}rails\b", re.IGNORECASE), "createsteamrails"),
    (re.compile(rf"\bbetter{PHRASE_SEP}end\b", re.IGNORECASE), "betterend"),
    (re.compile(rf"\bbetter{PHRASE_SEP}nether\b", re.IGNORECASE), "betternether"),
    (re.compile(rf"\bneo{PHRASE_SEP}forge\b", re.IGNORECASE), "neoforge"),
)

# 마인크래프트 생태계 고유명사와 추천 의도 단어는 불용어보다 우선 보존한다.
PROTECTED_TERMS = {
    "adventure", "ae2", "animation", "apocalypse", "applied", "ars",
    "automation", "backpack", "biome", "block", "boss", "botania", "bukkit",
    "betterend", "betternether", "bossbattle", "bossfight", "c2me", "cave",
    "challenging", "chat", "chatreports", "client", "clientside",
    "cobblemon", "combat", "create", "createsteamrails", "cursed", "datapack",
    "decoration", "dimension", "distanthorizons", "dungeon", "dynamicfps",
    "earlygame", "economy", "embeddium", "end", "endgame", "energistics",
    "engineering", "entity", "entityculling", "entitytexturefeatures",
    "equipment", "etf", "exploration", "fabric", "factory", "fantasy",
    "farmersdelight", "ferritecore", "food", "forge", "fps", "ftb", "gui",
    "hardcore", "highend", "horror", "hud", "immersive", "indium",
    "industrial", "inventory", "iris", "item", "kitchensink", "krypton",
    "kubejs", "lategame", "library", "lightweight", "lithium", "lowend",
    "magic", "management", "map", "medieval", "mekanism", "minigame",
    "minimap", "mmorpg", "mob", "modernfix", "modloader", "multiplayer",
    "neoforge", "nether", "noisium", "nvidium", "oculus", "oneblock", "ore",
    "openworld", "origin", "origins", "overworld", "paper",
    "performance", "phosphor", "pokemon", "progression", "purpur", "pve",
    "pvp", "qol", "quest", "questbook", "quilt", "refined",
    "refinedstorage", "resource", "resourcepack", "rpg", "server",
    "serverside", "shader", "shaderpack", "simplevoicechat", "singleplayer",
    "skilltree", "skyblock", "smp", "social", "sodium", "spigot", "starlight",
    "stoneblock", "storage", "structure", "superflat", "survival", "tech", "techtree",
    "technology", "terrain", "texture", "texturepack", "thermal",
    "thermalexpansion", "tomssimplestorage", "transportation", "utility",
    "vanilla", "vanillaplus", "visual", "voice", "voicechat", "vulkan",
    "vulkanmod", "waystone", "world", "worldgen", "worldmap",
    "xaerosminimap", "xaerosworldmap", "yung", "zombie",
}

# sklearn 기본 영어 불용어에 모드팩 설명에서 반복되는 홍보/문서용 표현을 더한다.
EN_STOPWORDS = {
    "minecraft", "modpack", "modpacks", "mod", "mods", "pack", "packs",
    "package", "packages", "include", "includes", "included", "including",
    "feature", "features", "content", "experience", "official", "version",
    "versions", "new", "support", "supports", "supported", "supporting",
    "need", "needs", "built", "designed", "based", "focused", "simple",
    "proper", "true", "best", "popular", "better", "play", "playing",
    "game", "mc", "modrinth", "discord", "player", "players", "thing",
    "things", "stuff", "config", "configuration", "menu", "add", "adds",
    "added", "adding", "summary", "details", "description", "information",
    "note", "important", "please", "click", "link", "links", "list",
    "previous", "current", "history", "historical", "start", "starts",
    "started", "starting", "mode", "modes", "use", "uses", "used", "using",
    "title", "section", "recommend", "recommends", "recommended",
    "recommendation", "install", "installs", "installed", "installing",
    "installation", "available", "default", "optional", "required",
    "compatible", "launcher", "profile", "settings", "issue", "issues",
    "normal", "release", "releases", "released", "report", "read",
    "work", "works", "working", "run", "runs", "running",
    "able", "allow", "allows", "try", "today", "thanks", "welcome",
    "smooth", "smoother", "improved", "improvement", "improvements",
    "quality", "life", "ram", "gb", "mb", "memory", "allocate", "allocated",
    "download", "downloads", "http", "https", "www", "com", "src", "href",
    "png", "jpg", "jpeg", "gif", "image", "images", "logo", "readme",
    "api", "button", "buttons", "changelog", "code", "css", "file", "files",
    "folder", "folders", "html", "javascript", "js", "json", "page", "pages",
    "port", "ports", "tab", "tabs", "toml", "xml", "yaml", "yml",
    "curseforge", "github", "wiki",
}
EN_STOPWORDS = (EN_STOPWORDS | set(SKLEARN_STOP_WORDS)) - PROTECTED_TERMS


# =========================
# HTML/Markdown 정리
# =========================

class HtmlToText(HTMLParser):
    """HTML 조각을 텍스트로 바꾸되 문단/목록 경계는 보존한다."""

    BLOCK_TAGS = {
        "address", "article", "aside", "blockquote", "br", "dd", "details",
        "div", "dl", "dt", "figcaption", "figure", "footer", "h1", "h2",
        "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "ol", "p",
        "pre", "section", "summary", "table", "tbody", "td", "tfoot", "th",
        "thead", "tr", "ul",
    }
    SKIP_TAGS = {
        "script", "style", "iframe", "svg", "video", "audio", "picture",
        "pre", "code", "kbd", "samp",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        # parts에 텍스트 조각과 필요한 줄바꿈을 순서대로 쌓아 마지막에 합친다.
        self.parts: list[str] = []
        # script/code 같은 제거 대상 태그 안쪽이면 handle_data가 텍스트를 추가하지 않게 깊이를 센다.
        self.skip_depth = 0

    def handle_starttag(self, tag, _attrs):
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            # 제거 대상 태그는 중첩될 수 있으므로 bool이 아니라 depth로 관리한다.
            self.skip_depth += 1
            return
        if tag == "img":
            # 이미지는 alt가 있어도 설명/본문 검색 신호로 쓰지 않는다.
            return
        if tag == "li":
            # 목록 항목은 앞 항목과 붙지 않게 줄바꿈과 목록 표시를 넣는다.
            self.parts.append("\n- ")
        elif tag in self.BLOCK_TAGS:
            # 블록 태그는 문장 경계를 보존하기 위해 줄바꿈으로 바꾼다.
            self.parts.append("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in self.SKIP_TAGS and self.skip_depth:
            # 제거 대상 태그가 끝나면 다시 일반 텍스트를 받을 수 있게 depth를 줄인다.
            self.skip_depth -= 1
            return
        if not self.skip_depth and tag in self.BLOCK_TAGS:
            # 닫는 블록 태그도 다음 텍스트와 붙지 않도록 줄바꿈을 넣는다.
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip_depth:
            # 제거 대상 태그 바깥의 실제 텍스트만 보존한다.
            self.parts.append(data)

    def get_text(self) -> str:
        # normalize_spacing이 뒤에서 공백을 정리하므로 여기서는 순서만 유지해 합친다.
        return "".join(self.parts)


def normalize_spacing(text: str) -> str:
    """문단 경계는 보존하고 줄 내부 공백만 정리한다."""
    lines = []
    for line in str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        # 탭/연속 공백은 한 칸으로 줄이되, 줄 자체는 남겨 문단 경계를 유지한다.
        lines.append(re.sub(r"[ \t\f\v]+", " ", line).strip())
    # 너무 많은 빈 줄은 두 줄까지만 남겨 이후 번역기가 문단 경계로 볼 수 있게 한다.
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def clean_text(text: str) -> str:
    """Modrinth description/body에 섞인 HTML, Markdown, 링크, 코드 노이즈를 제거한다."""
    # NaN이 문자열 "nan"으로 들어가지 않게 막고, &amp; 같은 HTML entity를 원래 문자로 되돌린다.
    text = html.unescape("" if pd.isna(text) else str(text))
    text = EMOJI_RE.sub(" ", text)

    # 코드/이미지/URL은 추천 의미보다 노이즈가 큰 경우가 많아 먼저 제거한다.
    # 블록 성격의 문법은 앞뒤 문장이 붙지 않도록 줄바꿈으로 치환한다.
    text = re.sub(r"<!--.*?-->", "\n", text, flags=re.DOTALL)
    text = re.sub(r"```.*?```|~~~.*?~~~", "\n", text, flags=re.DOTALL)
    text = re.sub(r"(?m)^\s{4,}\S.*$", "\n", text)
    text = re.sub(
        r"<(?:script|style|iframe|svg|video|audio|picture|pre|code|kbd|samp)\b[^>]*>.*?</(?:script|style|iframe|svg|video|audio|picture|pre|code|kbd|samp)>",
        "\n",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"\[!\[[^\]]*](?:\([^)]+\)|\[[^\]]+])]\([^)]+\)", "\n", text)
    text = re.sub(r"!\[[^\]]*]\([^)]+\)", "\n", text)
    text = re.sub(r"!\[[^\]]*]\[[^\]]+]", "\n", text)
    text = re.sub(r"\[([^\]]+)]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)]\[[^\]]+]", r"\1", text)
    text = re.sub(r"(?m)^\s*\[[^\]]+]:\s*\S+.*$", "\n", text)
    text = re.sub(r"<?(?:https?://[^\s<>)]+|www\.[^\s<>)]+|mailto:[^\s<>)]+)>?", " ", text)

    # 제목/목록의 텍스트는 남기고 Markdown 장식만 제거한다.
    # 목록 기호는 버리지만 목록 항목 문장은 추천 신호가 될 수 있어 남긴다.
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "\n", text)
    text = re.sub(r"(?m)^\s{0,3}>\s?", "", text)
    text = re.sub(r"(?m)^\s*[-*_]{3,}\s*$", "\n", text)
    text = re.sub(r"(?m)^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", "\n", text)
    text = re.sub(r"(?m)^\s*[-*+]\s+", "- ", text)
    text = re.sub(r"(?m)^\s*\d+[.)]\s+", "- ", text)
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", text)
    text = text.replace("|", " ")

    if re.search(r"</?[A-Za-z][^>]*>", text):
        # 정규식 정리 뒤에도 태그가 남아 있으면 HTMLParser로 텍스트만 다시 추출한다.
        parser = HtmlToText()
        parser.feed(text)
        parser.close()
        text = parser.get_text()

    return normalize_spacing(text)


def preprocess_english(text: str, nlp) -> list[str]:
    """spaCy로 영어 토큰을 나누고 품사/원형/불용어 기준으로 검색 토큰을 만든다."""
    text = html.unescape(str(text))
    # data pack, vanilla+처럼 여러 표기가 가능한 표현은 토큰화 전에 하나의 검색 토큰으로 통일한다.
    for pattern, replacement in DOMAIN_PHRASES:
        text = pattern.sub(replacement, text)
    # v2.0~2.2, v1.x 같은 모드팩 자체 버전/changelog 표기는 추천 신호가 약해 제거한다.
    text = PACK_VERSION_RE.sub(" ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"<?(?:https?://[^\s<>)]+|www\.[^\s<>)]+|mailto:[^\s<>)]+)>?", " ", text)
    text = re.sub(r"[`*_~<>{}\[\]()/\\|=:$;,!?\"']", " ", text)

    tokens: list[str] = []
    for token in nlp(text):
        # raw는 보호어/숫자/버전 판정에 쓰고, lemma는 일반 단어의 원형화 결과에 쓴다.
        raw = token.text.strip("._-").lower()
        if not raw or token.is_space or token.is_punct or token.like_url:
            continue

        parts = [raw] if NUMBER_RE.fullmatch(raw) else [part for part in re.split(r"[._-]+", raw) if part]
        if len(parts) > 1:
            # 하이픈/점으로 붙은 단어는 과분해하지 않고, 보호어로 등록된 조각만 살린다.
            for part in parts:
                if part in PROTECTED_TERMS:
                    tokens.append(part)
            continue

        term = parts[0] if parts else raw
        if term in PROTECTED_TERMS:
            tokens.append(term)
            continue
        # 게임 버전과 순수 숫자는 추천 의미보다 필터/메타데이터 성격이 강해 텍스트에서는 제외한다.
        if MINECRAFT_VERSION_RE.fullmatch(term):
            continue
        if NUMBER_RE.fullmatch(term):
            continue

        if token.pos_ not in KEEP_POS:
            continue
        lemma = (token.lemma_ or term).strip("._-").lower()
        if not TOKEN_RE.fullmatch(lemma):
            continue
        if lemma in EN_STOPWORDS:
            continue
        if len(lemma) <= 1:
            continue
        if NUMBER_RE.fullmatch(lemma):
            continue
        tokens.append(lemma)
    return tokens


def meta_tokens(row) -> list[str]:
    """tags/categories를 검색 가중치용 메타 토큰으로 만든다."""
    tokens: list[str] = []
    for column, prefix in (("tags", "tag"), ("categories", "category")):
        # tags/categories는 쉼표 문자열이므로 값별로 분리한다. 빈 값은 빈 리스트로 처리한다.
        raw_values = [] if pd.isna(row.get(column, "")) else str(row.get(column, "")).split(",")
        for raw in raw_values:
            value = raw.strip().lower()
            # 공백이 있는 카테고리는 하나의 메타 토큰이 되게 밑줄로 묶는다.
            value = re.sub(r"\s+", "_", value)
            # 메타 토큰은 TF-IDF에 들어가므로 검색 토큰에 쓸 수 있는 문자만 남긴다.
            value = re.sub(r"[^a-z0-9+#._-]+", "", value).strip("._-")
            if value:
                tokens.append(f"{prefix}:{value}")
    # tags와 categories에 같은 값이 반복될 수 있어 순서를 유지한 채 중복만 제거한다.
    return list(dict.fromkeys(tokens))


def insert_meta(tokens: list[str], meta: list[str]) -> list[str]:
    """메타 토큰을 시작, 일정 간격, 끝에 반복 삽입해 태그/카테고리 신호를 강화한다."""
    if not meta:
        # 메타 정보가 없는 row는 본문 토큰만 그대로 사용한다.
        return tokens

    # 시작 부분에 한 번 넣어 짧은 description에서도 메타 신호가 사라지지 않게 한다.
    output = meta[:]
    for index, token in enumerate(tokens, 1):
        output.append(token)
        if index % META_INTERVAL == 0:
            # 긴 본문에서는 META_INTERVAL마다 다시 넣어 뒤쪽 chunk에서도 태그/카테고리 신호가 보이게 한다.
            output.extend(meta)
    # 끝에도 한 번 넣어 마지막 토큰 구간에서 메타 신호가 약해지지 않게 한다.
    output.extend(meta)
    return output


def append_output_row(row, processed_description: str, complete: int, columns: list[str]) -> None:
    """row 하나의 결과를 최종 CSV에 바로 append해 중단 후 재시도할 수 있게 한다."""
    path = Path(OUTPUT_CSV)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = row.to_dict()
    record["description"] = processed_description
    record["complete"] = int(complete)

    # 한 row가 끝날 때마다 저장해야 긴 번역 작업 중단 후에도 성공한 row를 다시 처리하지 않는다.
    pd.DataFrame([record], columns=columns).to_csv(
        path,
        mode="a",
        header=not path.exists() or path.stat().st_size == 0,
        index=False,
        encoding="utf-8-sig",
    )


def main():
    df = pd.read_csv(INPUT_CSV)
    columns = list(df.columns) + (["complete"] if "complete" not in df.columns else [])

    # 이전 실행에서 complete=0/null로 끝난 실패 row는 재시도 대상이므로 출력 CSV에서 먼저 제거한다.
    # 이렇게 해두면 모델 생성/추천 단계가 "최종 CSV에는 성공 row만 남는다"는 단순한 전제를 가질 수 있다.
    complete_by_key: dict[str, bool] = {}
    output_path = Path(OUTPUT_CSV)
    if output_path.exists() and output_path.stat().st_size > 0:
        out_df = pd.read_csv(output_path)
        if "complete" not in out_df.columns:
            out_df["complete"] = 0
        complete_text = out_df["complete"].fillna("").astype(str).str.strip().str.lower()
        # 성공 row만 남기고 실패 row는 삭제한다. 실패 row가 남아 있으면 모델 생성 시 행 수가 꼬일 수 있다.
        complete_mask = complete_text.isin({"1", "1.0", "true", "yes", "y"})
        removed_rows = len(out_df) - int(complete_mask.sum())
        if removed_rows:
            out_df = out_df[complete_mask].copy()
            out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"[preprocess resume] removed incomplete rows={removed_rows}")
        for _idx, done_row in out_df.iterrows():
            # 남아 있는 출력 row는 모두 성공 row이므로, 같은 slug/url은 입력에서 다시 만나도 건너뛴다.
            slug = "" if pd.isna(done_row.get("slug", "")) else str(done_row.get("slug", ""))
            url = "" if pd.isna(done_row.get("url", "")) else str(done_row.get("url", ""))
            key = slug or url
            if key:
                complete_by_key[key] = True

    # 번역기는 실제 번역이 필요할 때 내부에서 모델을 로드한다. 여기서는 설정 객체만 준비한다.
    translator = TranslateGemmaTranslator(TranslateGemmaConfig(
        model_id=TRANSLATE_MODEL_ID,
        model_dir=TRANSLATE_MODEL_DIR,
        cache_path=TRANSLATION_CACHE,
        use_translation=USE_TRANSLATION,
        quantization=TRANSLATE_QUANTIZATION,
        model_dtype=MODEL_DTYPE,
        device_map=TRANSLATE_DEVICE_MAP,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=BNB_4BIT_COMPUTE_DTYPE,
        bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
        bnb_8bit_threshold=BNB_8BIT_THRESHOLD,
        max_input_tokens=MAX_TRANSLATE_INPUT_TOKENS,
        hard_max_input_tokens=HARD_MAX_TRANSLATE_INPUT_TOKENS,
        max_total_tokens=MAX_TRANSLATE_TOTAL_TOKENS,
        max_output_tokens=MAX_TRANSLATE_OUTPUT_TOKENS,
        min_output_tokens=MIN_TRANSLATE_OUTPUT_TOKENS,
        output_token_ratio=OUTPUT_TOKEN_RATIO,
        debug_on_failure=TRANSLATE_DEBUG_ON_FAILURE,
        debug_to_console=TRANSLATE_DEBUG_TO_CONSOLE,
        debug_dir=TRANSLATE_DEBUG_DIR,
        debug_max_chars=TRANSLATE_DEBUG_MAX_CHARS,
        audit_log=TRANSLATE_AUDIT_LOG,
        audit_dir=TRANSLATE_AUDIT_DIR,
        log_progress=TRANSLATE_LOG_PROGRESS,
        log_preview_chars=TRANSLATE_LOG_PREVIEW_CHARS,
    ))
    # parser/ner는 검색 토큰 생성에 쓰지 않으므로 꺼서 로드와 실행 비용을 줄인다.
    nlp = spacy.load(SPACY_MODEL_NAME, disable=["parser", "ner"])
    language_detector = (
        LanguageDetectorBuilder
        .from_all_languages()
        .with_preloaded_language_models()
        .build()
    ) if USE_TRANSLATION else None
    # 번역 캐시는 같은 원문/언어/모델 설정이면 다시 생성하지 않기 위한 저장소다.
    cache = load_translation_cache(TRANSLATION_CACHE)

    # 마지막 출력에서 처리 결과를 확인하기 위한 단순 카운터다.
    success_count = 0
    error_count = 0
    skipped_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        start = time.perf_counter()
        # resume 판단에는 원본 CSV index보다 slug/url이 안정적이다.
        slug = "" if pd.isna(row.get("slug", "")) else str(row.get("slug", ""))
        url = "" if pd.isna(row.get("url", "")) else str(row.get("url", ""))
        key = slug or url
        # 이미 성공 row가 출력 CSV에 있으면 같은 작업을 반복하지 않는다.
        if key and complete_by_key.get(key):
            skipped_count += 1
            continue

        # description/body 각각을 먼저 정리한 뒤, 둘이 한 문장처럼 붙지 않도록 빈 줄 두 개로 합친다.
        source_text = normalize_spacing("\n\n".join(
            part for part in (
                clean_text(row.get("description", "")),
                clean_text(row.get("body", "")),
            ) if part
        ))

        if not source_text:
            # 원문 텍스트가 전혀 없으면 만들 토큰도 없지만, 처리 자체는 끝난 row로 기록한다.
            append_output_row(row, "", 1, columns)
            if key:
                complete_by_key[key] = True
            success_count += 1
            continue

        try:
            english_text = source_text
            if USE_TRANSLATION and language_detector is not None:
                # Lingua가 반환한 enum을 ISO 639-1 문자열(en/ko/ja/zh 등)로 바꾼다.
                language = language_detector.detect_language_of(source_text)
                lang = "unknown" if language is None else language.iso_code_639_1.name.lower()
                # 영어/unknown은 번역하지 않는다. 비영어로 확정된 경우만 영어로 맞춘다.
                if lang not in {"en", "unknown"}:
                    english_text = translator.translate_text(
                        source_text,
                        lang,
                        "en",
                        cache,
                        label=f"row={idx} name={str(row.get('name', '')).strip()[:80]}",
                        raise_on_failure=True,
                    ) or source_text

            # 영어 텍스트를 검색 토큰으로 바꾸고, tags/categories 메타 토큰을 일정 간격으로 끼워 넣는다.
            tokens = insert_meta(preprocess_english(english_text, nlp), meta_tokens(row))
            # 성공 row는 description을 토큰 문자열로 교체하고 complete=1로 append한다.
            append_output_row(row, " ".join(tokens).strip(), 1, columns)
        except Exception as exc:
            # row 중 일부 chunk라도 실패하면 완료로 보지 않는다. 다음 실행에서 다시 처리하도록 complete=0으로 남긴다.
            error_count += 1
            append_output_row(row, "", 0, columns)
            if key:
                complete_by_key[key] = False
            save_translation_cache(cache, TRANSLATION_CACHE)
            print(f"[row failed] index={idx} error={exc}")
            continue

        if key:
            # 같은 실행 안에서 같은 key를 다시 만나도 중복 처리하지 않게 메모리 상태도 갱신한다.
            complete_by_key[key] = True
        success_count += 1

        if PRINT_ROW_TIMING:
            print(f"[row] index={idx} elapsed={time.perf_counter() - start:.1f}s")
        if success_count and success_count % CACHE_SAVE_EVERY == 0:
            # 긴 실행 중 프로세스가 죽어도 최근 번역 캐시를 잃지 않도록 주기적으로 저장한다.
            save_translation_cache(cache, TRANSLATION_CACHE)

    # 루프 종료 후 남은 캐시를 저장하고 전체 처리 요약을 출력한다.
    save_translation_cache(cache, TRANSLATION_CACHE)
    print(
        f"done: {OUTPUT_CSV} "
        f"success={success_count} errors={error_count} skipped={skipped_count}"
    )


if __name__ == "__main__":
    main()
