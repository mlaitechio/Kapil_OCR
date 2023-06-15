# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 18:13:27 2023

@author: kapil
"""

from PIL import Image, ImageSequence
import os

def tiff_to_pdf(tiff_path: str, file_type) -> str:
  if file_type == ".TIF":
    pdf_path = tiff_path.replace('.TIF', '.pdf')
    if not os.path.exists(tiff_path): raise Exception(f'{tiff_path} does not find.')
    image = Image.open(tiff_path)

    images = []
    for i, page in enumerate(ImageSequence.Iterator(image)):
        page = page.convert("RGB")
        images.append(page)
    if len(images) == 1:
        images[0].save(pdf_path)
    else:
        images[0].save(pdf_path, save_all=True,append_images=images[1:])
    return pdf_path




