from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
model_ocr = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
import warnings
warnings.filterwarnings("ignore")

def find_ocr(file):
    print(file)
    doc = DocumentFile.from_pdf(str(file))

    # Analyze
    result = model_ocr(doc)

#    result = model(doc[5:6])
    json_output = result.export()
    words_loc_normalized = []
    corpus = []
    for ii in range (len(json_output['pages'])):
      for d in json_output.values():
          num_blocks = len(d[ii]['blocks'])
      #    print(num_blocks)
  
          for k in range (num_blocks):
          
            row = d[ii]['blocks'][k]['lines']
     #       print(row)
            res = " ".join([k['value'] for item in row for k in item['words']])
         #   print(res)
            corpus.append(res)
            
    return " ".join(corpus)
