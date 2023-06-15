# -*- coding: utf-8 -*-
"""
Created on Thu May 25 22:37:10 2023

@author: kapil
"""

from pathlib import Path,PurePosixPath
from tif_to_pdf_convertor import tiff_to_pdf
from apply_doctr_ocr import find_ocr
import time

def write_to_textfile(filename):
  p = Path("temp/")
  p.mkdir(parents=True, exist_ok=True)

 # with p.open("temp."+fn, "w", encoding ="utf-8") as f:  
  with open(Path(p, filename, ".txt"), 'w') as f:
    f.write('readme')
    
if __name__ == "__main__":
    
    p = Path('demo-samples')
    for i in p.rglob(r'*.pdf'):
         j = Path(i)
         k = PurePosixPath(i)
         print(i,j.suffix, k)
       #  tiff_to_pdf(str(k), j.suffix)
         raw_ocr = find_ocr(k)
         write_to_textfile(k.name)
       
   
