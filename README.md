# Welcome to the SOCR repository !

## Requirements

 - A Linux or a Mac computer with CUDA 8 minimum
 - PyTorch
 - All the package specified in the ```requirements.txt``` file. You can install all the package by using : ```conda install --file requirements.txt```
 - OpenCV 3 for Python

__SOCR is incompatible with Python 2 and Windows__.

## How to recognize a page of text ?

### Command line

Please make sure that checkpoint_line.pth.tar and checkpoint_ocr.pth.tar is present.
Then run 
```python -m socr.recognizer [your files] [your folders]  ...```

### API

SOCR work with Pillow library to process the images.

```python
from PIL import Image
from socr import Recognizer as TextRecognizer

text_recognizer = TextRecognizer()
texts, lines_position = text_recognizer.recognize(Image.open("your_image.jpg"))
print("Full text : \n" + "\n".join(texts))
```


## How to train/test

### Settings up the database

Modify the file ```datasets.cfg``` so it correspond to your data-sets.

### How to train/test the line recognizer ?

Execute ```python -m socr.line_localizator``` to train the line recognizer.
Execute ```python -m socr.line_localizator --generateandexecute``` to test the line recognizer.

### How to train/test the ocr network ?

Execute ```python -m socr.text_recognizer``` to train the ocr network.
Execute ```python -m socr.text_recognizer --generateandexecute``` to test the ocr network.

You can use the three ```--model```, ```--optimizer```, ```--lr``` to specify the model and the opitmizer to use.

### Compare the SOCR Ocr networks with Ocropy models

The script ```compare_text_recognizer``` has been made to compare SOCR networks with Ocropy.
