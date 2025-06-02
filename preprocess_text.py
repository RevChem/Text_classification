import re
from nltk.tokenize import word_tokenize 
from pymorphy3 import MorphAnalyzer 
from stop_words import get_stop_words  
from sklearn.feature_extraction.text import CountVectorizer

RUSSIAN_STOP_WORDS = set(get_stop_words('ru')) 
morph = MorphAnalyzer()

def preprocess_text(text):
    if not isinstance(text, str):
        text = ""

    text = re.sub(r'@\w+\b', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^а-яё\s]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text, language='russian')
    lemmas = [
        morph.parse(t)[0].normal_form for t in tokens
        if t not in RUSSIAN_STOP_WORDS and len(t) >= 2
    ]

    return ' '.join(lemmas)

def build_vocab(texts, min_df=1):
    vectorizer = CountVectorizer(
        min_df=min_df,
        tokenizer=lambda x: x.split(),
        token_pattern=None
    )
    vectorizer.fit(texts)
    vocab = {word: idx+2 for idx, word in enumerate(vectorizer.get_feature_names_out())}
    vocab['<pad>'] = 0
    vocab['<unk>'] = 1
    return vocab

if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('dataset/train.csv')['text'][:100]
    print(build_vocab(df))