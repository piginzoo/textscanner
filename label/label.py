import numpy as np
import json


class ImageLabel:
    """
        Wrap the image and label data
        there are 2 types:
        - labelme format: like https://github.com/wkentaro/labelme/blob/master/examples/tutorial/apc2016_obj3.json
        - plaintext: like
                >>>
                你好，世界
                11,12,21,22,31,32,41,42,你
                ...
                <<<
        also, inside the class, the label need to convert to standard size(64x256)
    """

    # target_size is (W,H)
    def __init__(self, image, data, format):
        self.format = format
        self.image = image
        self.labels = self.load(data)

    def load(self, data):

        if self.format == "labelme":
            return self._load_labelme(data)

        if self.format == "plaintext":
            return self._load_plaintext(data)

        raise ValueError("Unknow label type:", self.format)

    # labelme json format reference: https://github.com/wkentaro/labelme/blob/master/examples/tutorial/apc2016_obj3.json
    def _load_labelme(self, data):

        assert type(data) == list

        data = "".join(data)

        image_labels = json.loads(data)
        shapes = image_labels['shapes']
        labels = []
        for s in shapes:
            label = s['label']
            points = s['points']
            labels.append(Label(label, points))
        return labels

    # format：
    #   你好，世界
    #   11,12,21,22,31,32,41,42,你
    #   11,12,21,22,31,32,41,42,好
    #   ....
    def _load_plaintext(self, data):

        assert type(data) == list

        # data[0], bypass the first line, which is the label strings

        # parse line #2 to end
        labels = []
        for i in range(1, len(data)):
            # "11,12,21,22,31,32,41,42,你"
            line = data[i]
            line = line.replace("\n","")

            line_data = line.split(",")
            points = line_data[:8]
            label = line_data[8]

            # "11,12,21,22,31,32,41,42" => [[11,12],[21,22],[31,32],[41,42]]
            points = [int(p.strip()) for p in points]
            points = np.array(points)
            points = np.reshape(points, (4, 2))
            labels.append(Label(label, points))
        return labels

    @property
    def bboxes(self):
        return np.array([l.bbox for l in self.labels])


class Label:
    """
        Single word label format:
            "label": "X",
            "points": [ [x1,y1],....,[xn,yn]]
    """

    def __init__(self, label, bbox):
        if type(bbox) == list:
            bbox = np.array(bbox)
        assert bbox.shape == (4, 2)
        self.bbox = bbox
        self.label = label


if __name__ == "__main__":
    f = open("data/test/a.json", encoding="utf-8")
    data = f.read()

    il = ImageLabel(data)
