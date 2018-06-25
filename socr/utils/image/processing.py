def binarize(image):
    return (image > 0.5).astype('unit8')