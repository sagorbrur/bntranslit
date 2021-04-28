# BNTRANSLIT
__BNTRANSLIT__ is a deep learning based transliteration app for Bangla word.

## Installation
`pip install bntranslit`

## Dependency
- pytorch 1.7.0+

## Usage

```py
from bntranslit import BNTransliteration

bntrans = BNTransliteration()

model_path = "bntranslit_model.pth"
word = "vaat"
output = bntrans.predict(model_path, word, topk=10)

```
