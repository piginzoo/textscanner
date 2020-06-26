# Branch explanation

This branch implements the Word_Formulation as single layer, 
which will caculate the possibility of all words by caculation of "Character Segmentation" and "Order Maps".

But this branch can not run in my GPU server, for it will bring OOM of GPU when I try to train it by a charsets of 4100 classes.

But, for small charset, it is a good way to handle the training.

So, if you want to train English or Number charsets, please use it.

## details

The model has a extra layer named "Word Formulation", so model will return word formuation. 
Then, it will be used for calculation for the accuracy with the labels,
which return from sequence:

```text
class ImageLabelLoader:
...
return images, {'character_segmentation': batch_cs,
                'order_map': batch_om,
                'localization_map': batch_lm,
                'word_formation':labels} <---- see, the labels
-----
class TextScannerModel:
...
metrics = {'word_formation': ['categorical_accuracy']}
...
def call(self, inputs, training=None):
    ...
    return {'character_segmentation': character_segmentation,
            'order_map': order_map,
            'localization_map': localization_map,
            'word_formation': word_formation}
```  

2020.6 piginzoo