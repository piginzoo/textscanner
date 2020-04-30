import numpy as np
import json
from ast import literal_eval
class ImageLabel:

    # labelme json format reference: https://github.com/wkentaro/labelme/blob/master/examples/tutorial/apc2016_obj3.json
    def __init__(self,image,labelme_json):
        self.image = image
        image_labels = json.loads(labelme_json)
        shapes = image_labels['shapes']
        self.labels = []
        for s in shapes:
            label = s['label']
            points = s['points']
            self.labels.append(Label(label,points))

    @property
    def bboxes(self):
        return np.array([l.bbox for l in self.labels])

class Label:
    """
        Single word label format:
            "label": "X",
            "points": [ [x1,y1],....,[xn,yn]]
    """

    def __init__(self,label,bbox):
        if type(bbox)==list:
            bbox = np.array(bbox)
        assert bbox.shape==(4,2)
        self.bbox = bbox
        self.label = label

if __name__=="__main__":
    f = open("data/test/a.json",encoding="utf-8")
    data = f.read()

    il = ImageLabel(data)