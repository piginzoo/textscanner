from utils import util
import numpy as np
import json, cv2
import logging

logger = logging.getLogger(__name__)


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
        more, ImageLabel is in charge of resizing to standard size (64x256).
    """

    def __init__(self, image, data, format, target_size):
        self.format = format
        self.image = cv2.resize(image, target_size) # do the standard resizing

        self.target_size = target_size  # (W,H)
        self.orignal_size = (image.shape[1], image.shape[0])  # (W,H)

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
            points = util.resize_bboxes(points, original_size=self.orignal_size, target_size=self.target_size)
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
            line = line.replace("\n", "")

            line_data = line.split(",")

            points = line_data[:8]
            label = line_data[8]

            # "11,12,21,22,31,32,41,42" => [[11,12],[21,22],[31,32],[41,42]]
            points = [int(p.strip()) for p in points]
            points = np.array(points)
            points = np.reshape(points, (4, 2))

            # adjust all bboxes' coordinators
            points = util.resize_bboxes(points, original_size=self.orignal_size, target_size=self.target_size)

            # logger.debug("resized bbox:%r", points)

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
