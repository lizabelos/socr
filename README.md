# Structured OCR

## About

Structured OCR is a line localization and text extraction tools using Deep Learning with PyTorch.

## Requirements

 - A Linux or a Mac computer with CUDA 8 minimum
 - PyTorch
 - All the package specified in the ```requirements.txt``` file. You can install all the package by using : ```conda install --file requirements.txt```
 - OpenCV 3 for Python. You can install it using ```conda install --channel https://conda.anaconda.org/menpo opencv3```

__SOCR is incompatible with Python 2 and Windows__.

## Compilation

The compilation is not needed to use SOCR. But it will be more slower without.
To compile it, run : 

```python3 setup.py build_ext --inplace```

## How to recognize a page of text ?

### Command line

Please make sure that ```checkpoint_line.pth.tar``` and ```checkpoint_ocr.pth.tar``` is present under the ```checkpoints``` filder.
Then run 
```python -m socr.recognizer [your files] [your folders]  ...```

### API

SOCR work with Numpy library to process the images.

```python
from socr.line_localizator import LineLocalizator
from socr.text_recognizer import TextRecognizer

line_localizator = LineLocalizator()
text_recognizer = TextRecognizer()

line_localizator.eval()
text_recognizer.eval()

lines, positions = line_localizator.extract(original_image, resized)
texts = text_recognizer.recognize(lines)

print("Full text : \n" + "\n".join(texts))
```


## How to train/test

### Settings up the database

Modify the file ```datasets.cfg``` so it correspond to your data-sets.

```cfg
[DocumentGenerator]
for = Document
type = DocumentGenerator
train = yes
test = yes

[ICDAR]
for = Document
type = ICDAR-Baseline
train = /dataset/icdar/img/training

[LineGenerator]
for = Line
type = LineGenerator
train = yes
test = no

[IAM]
for = Line
type = IAM
train = /dataset/iam-line/train
test = /dataset/iam-line/test

[IAM-One-Line]
for = Line
type = IAM-One-Line
train = /dataset/iam-one-line/train
test = /dataset/iam-one-line/test

[IAM-Washington]
for = Line
type = IAM-Washington
train = /dataset/iam-washington/

```

### How to train/test the line recognizer ?

Execute ```python -m socr.line_localizator``` to train the line recognizer.
Execute ```python -m socr.line_localizator --generateandexecute``` to test the line recognizer.

### How to train/test the ocr network ?

Execute ```python -m socr.text_recognizer``` to train the ocr network.
Execute ```python -m socr.text_recognizer --generateandexecute``` to test the ocr network.

You can use the three ```--model```, ```--optimizer```, ```--lr``` to specify the model and the opitmizer to use.

### Compare the SOCR Ocr networks with Ocropy models

The script ```compare_text_recognizer``` has been made to compare SOCR networks with Ocropy.
