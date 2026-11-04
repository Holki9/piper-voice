# 🎙️ Персональный офлайн-голос Piper TTS (Русский язык)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Holki9/piper-voice/blob/main/Train_Piper_Voice.ipynb)

Этот репозиторий содержит базу аудиозаписей, скрипты нарезки и блокнот для быстрого дообучения (Fine-Tuning) русской нейросетевой модели Piper TTS на базе собственного дикторского голоса.

---

### 🚀 Быстрый старт в Google Colab (в 1 клик)

1. Нажмите на бейдж выше: **[Open In Colab](https://colab.research.google.com/github/Holki9/piper-voice/blob/main/Train_Piper_Voice.ipynb)**
2. В верхнем меню выберите: `Среда выполнения` -> `Сменить тип среды` -> **T4 GPU**.
3. Нажмите `Среда выполнения` -> **Выполнить всё** (`Ctrl + F9`).
4. Примерно через 30 минут модель `my_voice.onnx` будет обучена, протестирована и автоматически скачана на ваше устройство!

---

### 📂 Структура репозитория
* `raw_audio/` — оригинальные высококачественные студийные записи с телефона.
* `sentences.json` — 50 эталонных фраз на русском языке.
* `Train_Piper_Voice.ipynb` — готовый интерактивный пайплайн обучения.
