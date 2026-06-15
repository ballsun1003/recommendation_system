# 🎮 Mine Your Pack — Minecraft Modpack Recommendation System

Modrinth API에서 수집한 모드팩 데이터를 기반으로, 사용자가 입력한 설명과 가장 유사한 모드팩을 추천해주는 시스템입니다.  
TF-IDF / Word2Vec 기반 유사도 계산과 PyQt5 GUI를 사용하며, 영어 외 언어 입력 시 Gemma 번역 모델을 통해 자동 번역을 지원합니다.

---

## 📁 Project Structure

```
recommendation_system/
├── modrinth_dataset.py          # Modrinth API로 모드팩 데이터 수집
├── generate_model.py            # TF-IDF / Word2Vec 모델 학습 및 저장
├── preprocessor.py              # 텍스트 전처리 (spaCy 영어 모델 사용)
├── recommend_modpacks.py        # 추천 로직 (메인 실행 파일)
├── translategemma_translator.py # 비영어 입력 자동 번역 (Gemma 모델)
├── modpack_recommendation.ui    # PyQt5 UI 파일
└── requirements.txt
```

---

## ⚙️ Installation

### 1. 저장소 클론

```bash
git clone https://github.com/kimsu66/recommendation_system.git
cd recommendation_system
```

### 2. Python 환경 준비

Python **3.10** 이상을 권장합니다.  
가상환경 사용을 추천합니다.

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. 패키지 설치

> ⚠️ **CUDA 환경 여부에 따라 설치 방법이 다릅니다. 반드시 아래를 확인하세요.**

---

#### ✅ CUDA GPU가 있는 경우 (권장)

`requirements.txt`에는 **CUDA 12.8** 기준 PyTorch(`torch==2.11.0+cu128`)가 명시되어 있습니다.  
본인의 CUDA 버전이 12.8이라면 그대로 설치해도 됩니다.

```bash
pip install -r requirements.txt
```

CUDA 버전이 다를 경우 (예: CUDA 11.8, 12.1 등), PyTorch 공식 사이트에서 버전에 맞는 명령어를 확인하세요:  
👉 https://pytorch.org/get-started/locally/

예시 (CUDA 12.1):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

#### 🖥️ CPU만 있는 경우 (CUDA 없음)

`requirements.txt`를 그대로 설치하면 CUDA 버전의 PyTorch가 설치되어 오류가 발생할 수 있습니다.  
아래와 같이 CPU 버전 PyTorch를 **먼저** 설치한 뒤 나머지 패키지를 설치하세요.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

> ⚠️ CPU 환경에서는 번역 모델(Gemma) 실행 속도가 매우 느릴 수 있습니다.

---

### 4. spaCy 영어 모델 다운로드

이 프로젝트는 영어 텍스트 전처리에 **spaCy**의 `en_core_web_sm` 모델을 사용합니다.  
패키지 설치 후 반드시 아래 명령어를 **한 번** 실행해야 프로그램이 정상적으로 시작됩니다.

```bash
python -m spacy download en_core_web_sm
```

---

## 🤗 Hugging Face 설정 (비영어 입력 사용 시)

추천기에 **영어 이외의 언어**를 입력하면, 내부적으로 **Gemma 기반 번역 모델**을 사용하여 영어로 자동 번역합니다.  
이 번역 모델은 약 **3~4GB** 용량으로, 처음 사용 시 자동으로 다운로드됩니다.

단, 다운로드 전에 아래 두 가지를 완료해야 합니다.

### 1. Hugging Face 모델 접근 승인

아래 링크에서 모델 사용 승인을 요청하세요. (버튼 클릭만으로 즉시 승인됩니다.)

👉 https://huggingface.co/google/gemma-4b-it (또는 해당 translate 모델 페이지)

### 2. Hugging Face CLI 로그인

```bash
pip install huggingface_hub
huggingface-cli login
```

명령 실행 후 Hugging Face 계정의 **Access Token**을 입력하면 됩니다.  
Token은 https://huggingface.co/settings/tokens 에서 발급받을 수 있습니다.

> 영어로만 사용할 경우 이 과정은 생략해도 됩니다.

---

## 🚀 Usage

### Step 1. 데이터 수집

```bash
python modrinth_dataset.py
```

Modrinth API에서 모드팩 데이터를 수집하여 CSV로 저장합니다.

### Step 2. 모델 생성

```bash
python generate_model.py
```

TF-IDF 벡터라이저와 Word2Vec 모델을 학습하고 저장합니다.

### Step 3. 추천 시스템 실행

```bash
python recommend_modpacks.py
```

PyQt5 GUI가 실행되며, 원하는 모드팩 설명을 입력하면 유사한 모드팩을 추천합니다.

---

## 📦 Key Dependencies

| 패키지 | 용도 |
|---|---|
| `spacy` + `en_core_web_sm` | 영어 텍스트 전처리 |
| `scikit-learn` | TF-IDF 벡터화 |
| `gensim` | Word2Vec 모델 |
| `transformers` + `torch` | Gemma 번역 모델 |
| `PyQt5` | GUI |
| `huggingface_hub` | 모델 다운로드 인증 |

---

## ❓ Troubleshooting

**Q. `OSError: [E050] Can't find model 'en_core_web_sm'` 오류가 발생해요**  
→ `python -m spacy download en_core_web_sm` 명령어를 실행하세요.

**Q. `torch` 설치 시 CUDA 관련 오류가 발생해요**  
→ 본인의 CUDA 버전에 맞는 PyTorch를 https://pytorch.org 에서 확인 후 별도 설치하세요. CPU만 있다면 `--index-url https://download.pytorch.org/whl/cpu` 옵션을 사용하세요.

**Q. 번역 모델 다운로드 중 인증 오류가 발생해요**  
→ `huggingface-cli login` 으로 로그인했는지, 그리고 Hugging Face 사이트에서 해당 모델 접근 승인을 받았는지 확인하세요.
