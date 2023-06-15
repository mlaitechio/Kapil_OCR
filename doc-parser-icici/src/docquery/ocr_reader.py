import abc
import logging
from typing import Any, List, Optional, Tuple

import numpy as np
from pydantic.fields import ModelField


class NoOCRReaderFound(Exception):
    def __init__(self, e):
        self.e = e

    def __str__(self):
        return f"Could not load OCR Reader: {self.e}"


OCR_AVAILABLE = {
    "tesseract": False,
    "easyocr": False,
    "dummy": True,
    "microsoft" : False,
    "paddlepaddle" : False,
    "docTr" : False
}

try:
    import pytesseract  # noqa

    pytesseract.get_tesseract_version()
    OCR_AVAILABLE["tesseract"] = True
except ImportError:
    pass
except pytesseract.TesseractNotFoundError as e:
    logging.warning("Unable to find tesseract: %s." % (e))
    pass

try:
    import easyocr  # noqa

    OCR_AVAILABLE["easyocr"] = True
except ImportError:
    pass

try:

    OCR_AVAILABLE["microsoft"] = True
except ImportError:
    pass

try:

    OCR_AVAILABLE["paddlepaddle"] = True
except ImportError:
    pass

try:

    OCR_AVAILABLE["docTr"] = True
except ImportError:
    pass


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class OCRReader(metaclass=SingletonMeta):
    def __init__(self):
        # TODO: add device here
        self._check_if_available()

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, field: ModelField):
        if not isinstance(v, cls):
            raise TypeError("Invalid value")
        return v

    @abc.abstractmethod
    def apply_ocr(self, image: "Image.Image") -> Tuple[List[Any], List[List[int]]]:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def _check_if_available():
        raise NotImplementedError

class PaddleReader(OCRReader):
    def __init__(self):
        super().__init__()

    def apply_ocr(self, image: "Image.Image") -> Tuple[List[str], List[List[int]]]:
        """
        Applies Tesseract on a document image, and returns recognized words + normalized bounding boxes.
        This was derived from LayoutLM preprocessing code in Huggingface's Transformers library.
        """
        from paddleocr import PaddleOCR, draw_ocr
        import cv2
    
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
        image = np.asarray(image)           # GIVE IMAGE PATH HERE

        result = ocr.ocr(image, cls=True)
        result = result[0]
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        
        words = [t for t in txts]
        
        words_bbox = [[item[0][0],item[0][1],item[2][0],item[2][1]] for item in boxes]
        print(words, words_bbox)
        return words, words_bbox


    @staticmethod
    def _check_if_available():
        if not OCR_AVAILABLE["paddlepaddle"]:
            raise NoOCRReaderFound(
                "Unable to use pytesseract (OCR will be unavailable). Install tesseract to process images with OCR."
            )
            
class MsOCRReader(OCRReader):
    def __init__(self):
        super().__init__()

    def apply_ocr(self, image: "Image.Image") -> Tuple[List[str], List[List[int]]]:
        """
        Applies Tesseract on a document image, and returns recognized words + normalized bounding boxes.
        This was derived from LayoutLM preprocessing code in Huggingface's Transformers library.
        """
        import cv2,requests,time,json,glob
        import numpy as np
        from pathlib import Path    
        subscription_key = "cd9de642c81c46ddbe6509ad84c9b621"#"8b77eadc000640d4a292194794995dc6"#"94e072db7b7f451f82252fa11686c1e5" #"cd9de642c81c46ddbe6509ad84c9b621"
        text_recognition_url = "https://centralindia.api.cognitive.microsoft.com/vision/v2.0/read/core/asyncBatchAnalyze"
        image_url = np.asarray(image)           # GIVE IMAGE PATH HERE

        success, encoded_image = cv2.imencode('.jpg', image_url)
        image_data = encoded_image.tobytes()    								
        headers = {'Ocp-Apim-Subscription-Key': subscription_key,
                 'Content-Type': 'application/octet-stream'}
        params = {'visualFeatures': 'Categories,Description,Color','language': 'unk'}
        response = requests.post(text_recognition_url, headers=headers, params=params, data=image_data)
        response.raise_for_status()
          
        operation_url = response.headers["Operation-Location"]
          
        analysis = {}
        poll = True
        raw = []
        while (poll):
              response_final = requests.get(operation_url, headers=headers)
              analysis = response_final.json()
              time.sleep(1)
              if ("recognitionResults" in analysis):
                  poll= False 
              if ("status" in analysis and analysis['status'] == 'Failed'):
                  poll= False
          
        polygons=[]
        if ("recognitionResults" in analysis):
              polygons = [(line["boundingBox"], line["text"])
                          for line in analysis["recognitionResults"][0]["lines"]]
          
          
        words,words_bbox = [],[]
        for polygon in polygons:
              vertices = [(polygon[0][i], polygon[0][i+1])
                          for i in range(0, len(polygon[0]), 2)]
              start_pt,end_pt = vertices[0], vertices[2]
              text     = polygon[1].upper()
              words.append((text))
              words_bbox.append([start_pt[0],start_pt[1],end_pt[0],end_pt[1]])
              
              
        print(words, words_bbox)
          
        return words, words_bbox

    @staticmethod
    def _check_if_available():
        if not OCR_AVAILABLE["microsoft"]:
            raise NoOCRReaderFound(
                "Unable to use pytesseract (OCR will be unavailable). Install tesseract to process images with OCR."
            )

class TesseractReader(OCRReader):
    def __init__(self):
        super().__init__()

    def apply_ocr(self, image: "Image.Image") -> Tuple[List[str], List[List[int]]]:
        """
        Applies Tesseract on a document image, and returns recognized words + normalized bounding boxes.
        This was derived from LayoutLM preprocessing code in Huggingface's Transformers library.
        """
        data = pytesseract.image_to_data(image, output_type="dict")
        words, left, top, width, height = data["text"], data["left"], data["top"], data["width"], data["height"]

        # filter empty words and corresponding coordinates
        irrelevant_indices = set(idx for idx, word in enumerate(words) if not word.strip())
        words = [word for idx, word in enumerate(words) if idx not in irrelevant_indices]
        left = [coord for idx, coord in enumerate(left) if idx not in irrelevant_indices]
        top = [coord for idx, coord in enumerate(top) if idx not in irrelevant_indices]
        width = [coord for idx, coord in enumerate(width) if idx not in irrelevant_indices]
        height = [coord for idx, coord in enumerate(height) if idx not in irrelevant_indices]

        # turn coordinates into (left, top, left+width, top+height) format
        actual_boxes = [[x, y, x + w, y + h] for x, y, w, h in zip(left, top, width, height)]

        return words, actual_boxes

    @staticmethod
    def _check_if_available():
        if not OCR_AVAILABLE["tesseract"]:
            raise NoOCRReaderFound(
                "Unable to use pytesseract (OCR will be unavailable). Install tesseract to process images with OCR."
            )

class EasyOCRReader(OCRReader):
    def __init__(self):
        super().__init__()
        self.reader = None

    def apply_ocr(self, image: "Image.Image") -> Tuple[List[str], List[List[int]]]:
        """Applies Easy OCR on a document image, and returns recognized words + normalized bounding boxes."""
        if not self.reader:
            # TODO: expose language currently setting to english
            self.reader = easyocr.Reader(["en"])  # TODO: device here example: gpu=self.device > -1)

        # apply OCR
        data = self.reader.readtext(np.array(image))
        boxes, words, acc = list(map(list, zip(*data)))

        # filter empty words and corresponding coordinates
        irrelevant_indices = set(idx for idx, word in enumerate(words) if not word.strip())
        words = [word for idx, word in enumerate(words) if idx not in irrelevant_indices]
        boxes = [coords for idx, coords in enumerate(boxes) if idx not in irrelevant_indices]
        print("boxes", words)
        # turn coordinates into (left, top, left+width, top+height) format
        actual_boxes = [tl + br for tl, tr, br, bl in boxes]
        print("actual_boxes", actual_boxes)
        return words, actual_boxes

    @staticmethod
    def _check_if_available():
        if not OCR_AVAILABLE["easyocr"]:
            raise NoOCRReaderFound(
                "Unable to use easyocr (OCR will be unavailable). Install easyocr to process images with OCR."
            )


class DummyOCRReader(OCRReader):
    def __init__(self):
        super().__init__()
        self.reader = None

    def apply_ocr(self, image: "Image.Image") -> Tuple[(List[str], List[List[int]])]:
        raise NoOCRReaderFound("Unable to find any OCR reader and OCR extraction was requested")

    @staticmethod
    def _check_if_available():
        logging.warning("Falling back to a dummy OCR reader since none were found.")


OCR_MAPPING = {
    "tesseract": TesseractReader,
    "easyocr": EasyOCRReader,
    "dummy": DummyOCRReader,
    "microsoft" : MsOCRReader,
    "paddlepaddle" : PaddleReader
}


def get_ocr_reader(ocr_reader_name: Optional[str] = None):
    if not ocr_reader_name:
        for name, reader in OCR_MAPPING.items():
            if OCR_AVAILABLE[name]:
                return reader()

    if ocr_reader_name in OCR_MAPPING.keys():
        if OCR_AVAILABLE[ocr_reader_name]:
            return OCR_MAPPING[ocr_reader_name]()
        else:
            raise NoOCRReaderFound(f"Failed to load: {ocr_reader_name} Please make sure its installed correctly.")
    else:
        raise NoOCRReaderFound(
            f"Failed to find: {ocr_reader_name} in the available ocr libraries. The choices are: {list(OCR_MAPPING.keys())}"
        )
