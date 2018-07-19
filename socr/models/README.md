# Structured OCR

## How to create a model ?

Create a python file with the wanted model name, for example ```my_model.py```.

This file must contains a ```MyModel``` class which need to herit from ```Model``` or ```ConvolutionalModel```.

You must add your model to the ```__init__.py``` file before you can use it.