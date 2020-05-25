import numpy as np
import json


class ImageLabel:

    def __init__(self, image, data, format):
        self.image = image
        self.format = format
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
            self.labels.append(Label(label, points))
        return labels

    # 格式：
    #   你好，世界
    #   11,12,21,22,31,32,41,42,你
    #   11,12,21,22,31,32,41,42,好
    #   ....
    def _load_plaintext(self, data):

        assert type(data) == list

        # data[0],第一行，是整个字符串，忽略

        # 第2行至结束，解析
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
