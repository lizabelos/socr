# Structured OCR

## About

Structured OCR is a line localization and text extraction tools using Deep Learning with PyTorch.

## Requirements

**INFORMATION** : You can use ```python3 setup.py install_requirements``` to install all the requirements in one time !

 - A Linux or a Mac computer with CUDA 8 minimum
 - Python 3 with Anaconda
 - PyTorch 0.4 : ```conda install pytorch -c pytorch```
 - GCC/G++ version 5 (even if it's not your default compiler)
 - All the package specified in the ```requirements.txt``` file. You can install all the package by using : ```pip install -r requirements.txt```
 - OpenCV 3 for Python. You can install it using ```conda install --channel https://conda.anaconda.org/menpo opencv3```

Only on Windows :
 - Microsoft Visual C++ 14.0 (2015) is required. Get it with "Microsoft Visual C++ Build Tools": https://www.microsoft.com/en-us/download/details.aspx?id=48159
 - If you have problems with cupy installation with pip, install it with conda command instead : ```conda install cupy```
 - If you have conflicts between OpenCV and Zict, install OpenCV in the ```defaults``` channel using ```conda install opencv```
 - On Windows, use ```python -m socr``` instead of ```./launch```
 
## Compilation

To compile, run : 

```
python3 setup.py build_ext --inplace
```

The first run of the program will download resources, and install ctc wrapper and sru.

## How to recognize a page of text ?

### Command line

Please make sure that ```checkpoint_line.pth.tar``` and ```checkpoint_ocr.pth.tar``` is present under the ```checkpoints``` filder.
Then run 
```./launch recognizer [your files] [your folders]  ...```

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

Execute ```./launch line``` to train the line recognizer.
Execute ```./launch line --generateandexecute``` to test the line recognizer.

Use ```./launch line --help``` for more help.

### How to train/test the ocr network ?

Execute ```./launch text``` to train the ocr network.
Execute ```./launch text --evaluate [path]``` to test the ocr network. The result are placed under the ```results``` folder.

Use ```./launch text --help``` for more help.

### Command line arguments

You can select a model using the ```--model [MODEL]``` argument, and give a custom name to a model by using ```--name``` argument.
When ```CTRL-C``` is pressed, the program will ask you to save the weight or not. The weights are saved under the checkpoints folder, with the given name as argument, or with the model name if no name is specified.
The programe will automatically load the good model.

In can of problem, a backup of the weights are made with the extension ```.autosave``` under the ```checkpoints``` folder.

The ```--lr [lr]``` option permit to select the initial learning rate. Use ```--overlr``` to override the current learning rate.

The ```--bs``` option permit to specify a batch size.

### Multi-GPU

Use ```CUDA_VISIBLE_DEVICES=0,1``` to select the available GPU. SOCR will use all available GPU if you are training, or one GPU if you only evaluate.

## Make a custom model

You can make a custom model by placing it under the ```models``` folder. See ```dilatation_gru_network.py``` and ```dhSegment.py``` for example.
Each model has to be declared into the ```__init__.py``` file under the ```models``` folder.
