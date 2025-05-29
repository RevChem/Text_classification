import re
# Набор строковых констант. string.punctuation - все знаки пунктуации
import string
from nltk.tokenize import word_tokenize as PUNCT_WORD_TOKENIZER  # Токенизатор, разбивает текст на слова
from pymorphy3 import MorphAnalyzer  # Библиотека для морфологического анализа русского языка
from stop_words import get_stop_words  
from collections import Counter

RUSSIAN_STOP_WORDS = set(get_stop_words('ru')) 

def preprocess_text(text):
    text = re.sub(r'@\w+\b', '', text)
    text = re.sub(r'[0-9]+', '', text.lower())
    text = ''.join([ch if ch not in string.punctuation else ' ' for ch in text])
    text = re.sub(r'[^а-яё\s]', ' ', text, flags=re.IGNORECASE)
    
    tokens = PUNCT_WORD_TOKENIZER(text)

    norm_tokens = [
        MorphAnalyzer().parse(token)[0].normal_form
        for token in tokens
        if token not in RUSSIAN_STOP_WORDS and len(token)>0
    ]
    return ' '.join(norm_tokens)

def build_vocab(texts, word_freq=3):
    vocab = {'<pad>': 0, '<unk>': 1}
    idx = 2
    all_tokens = []

    for text in texts:
        tokens = PUNCT_WORD_TOKENIZER(preprocess_text(text))
        all_tokens.extend(tokens)

    freq = Counter(all_tokens)

    for token, count in freq.items():
        if count >=word_freq:
            vocab[token] = idx
            idx += 1

    return vocab
