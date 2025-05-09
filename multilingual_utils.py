import pandas as pd
from iso639 import languages
from typing import TypedDict

class Language(TypedDict):
    code: str
    language: str

def get_languages(file, lang_column) -> Language:
    codes = pd.read_csv(file)[lang_column].unique().tolist()
    results:Language = {}
    for code in codes:
        results[code] = languages.get(alpha2=code).name
    return results
        

print(get_languages(file="data/test_multilingual.csv", lang_column="lang"))
# {'tr': 'Turkish', 
# 'ru': 'Russian',
#  'it': 'Italian',
#   'fr': 'French', 
#   'pt': 'Portuguese',
#    'es': 'Spanish'}

