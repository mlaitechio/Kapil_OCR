# -*- coding: utf-8 -*-


import gradio as gr

import requests
from PIL import Image
from torchvision import transforms

import torch

from PIL import Image
from pathlib import Path
import streamlit as st
import cv2
import numpy as np
import torch
import warnings
from docquery import document, pipeline
from collections import OrderedDict
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
import matplotlib.pyplot as plt
import itertools
from statistics import mode
import pandas as pd
from paddleocr import PaddleOCR, draw_ocr
from fastai.vision.all import *

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

warnings.filterwarnings("ignore", category=FutureWarning)

ocr = PaddleOCR(use_angle_cls=True, lang='en')

layout_model = pipeline('document-question-answering', model="impira/layoutlm-document-qa")
model = torch.hub.load('ultralytics/yolov5', 'custom', path= "blob_detector.pt", force_reload=False)  # local model inv_250.pt

model_ocr = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

learn_inf = load_learner('doc_classifier.pkl')
learn_inf.dls.vocab    

# Download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

def grouper(iterable):
        prev = None
        group = []
        for p1,p2,p3,p4,p5 in iterable:
            if prev is None or p2 - prev <= 15:
                group.append((p1,p2,p3,p4,p5))
            else:
                yield group
                group = [(p1,p2,p3,p4,p5)]
            prev = p2
        if group:
            yield group
            
def get_ocr(table, words_loc_actual):
    result = ocr.ocr(table, cls=True)
    result = result[0]
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    
    
    bbox = [(item[0][0],item[0][1],item[2][0],item[2][1], txt) for item,txt in zip(boxes,txts)]
    extracted = OrderedDict()
    
#    if ocr_type == "docTr":   
#        bbox = [(item1[0],item1[1],item2[0],item2[1], txt) for txt,item1,item2 in words_loc_actual]
    print("============bbox============")
    print(bbox)
    p = dict(enumerate(grouper(bbox), 1))
    print("============ppppppppppppppp============")
    print(p)
    tot_rect, res, final_dict = [], [], {}
     
    for k,v in p.items():
        if len(v) >= 1:
            
            v = list(sorted(v, key=lambda x: x[0]))
            v= [(x1,y1,x2,y2,t) for x1,y1,x2,y2,t in v if x2-x1 >20]
            tot_rect.append(len(v))
            final_dict[k] = v
            
    
    avg_v = 2#(np.mean(tot_rect));  #  print("average cols", avg_v)

    for k,v in final_dict.items():
      color = list(np.random.random(size=3) * 256)
      if len(v) >= avg_v:
  #      print("---->>>>", v)
        row = []
        for i in range (len(v)):
            
          key_x1, key_y1, key_x2, key_y2 = int(v[i][0]), int(v[i][1]), int(v[i][2]), int(v[i][3])
     #     cv2.rectangle(im2arr, (key_x1, key_y1), (key_x2, key_y2), color, 2)
          row.append(v[i][4])
          
        res.append(row)
    #print(res)
    
    for item in res:
        is_num = item[0].replace(",","").isdigit()
        if not is_num:
            extracted[item[0].lower()] = [ item[i] for i in range(1, len(item))]
          
    output = dict(extracted)
    # print("output", output, len(output))
    tot_len = [] 
    for k,v in output.items():
        tot_len.append(len(v))
    
    try:
        avg_len = mode(tot_len);    #print("average len", avg_len)
        ref_output = {k : v for k,v in output.items() if len(v) == avg_len}
   # print("ref_output",ref_output,len(ref_output))    
    
        df= pd.DataFrame(ref_output)
        df1_transposed = df.T
       
        print("raw", df1_transposed)
        return df1_transposed
    
     #   print("final", res1)
     
    except:
        pass

        

def apply_ocr(file_path):
    name = Path(file_path).stem
    print(name)
    doc = DocumentFile.from_images(file_path)
    result = model_ocr(doc)
    
    synthetic_pages = result.synthesize()
    plt.imshow(synthetic_pages[0]); plt.axis('off'); plt.show()
    
    json_output = result.export()
            
    words_loc_normalized = []
    for d in json_output.values():
        num_blocks = len(d[0]['blocks']);#print(num_blocks)
        
        for i in range(num_blocks):
          num_lines = len(d[0]['blocks'][i]['lines']) ;#print(i, num_lines)
    
          for j in range(num_lines):
              row = ' '.join([word['value'] for word in d[0]['blocks'][i]['lines'][j]['words']])
              
         #     myfile.write("%s\n" % row)
    
              words_loc_normalized.append([(word['value'],word['geometry']) for word in d[0]['blocks'][i]['lines'][j]['words']])
        #  st.write(row)
    # flatten list of lists
    words_loc_normalized = list(itertools.chain(*words_loc_normalized))
    
    # convert relative pixel coordinates to actual image coordinates
    h, w = (json_output['pages'][0]['dimensions']); #print(w,h)
    words_loc_actual = list(map(lambda x: (x[0],(int(x[1][0][0]*w),int(x[1][0][1]*h)),(int(x[1][1][0]*w),int(x[1][1][1]*h))) , words_loc_normalized))
    
    print(words_loc_actual)
    return words_loc_actual

def predict(inp):
  #  img2 = np.asarray(inp)
    cv2.imwrite("dummy.jpg", inp)  
    doc = document.load_document("dummy.jpg")
    im = [inp]
    pred,pred_idx,probs = learn_inf.predict(inp)
    
    res = OrderedDict()
    df1_transposed = {}
    predicted = inp.copy()

    if pred == "Annex to Ops":
        kws = ["application number?","iLENS ID","loan ID","amount?","tenure?"]
        for q in kws:
            if layout_model(question=q, **doc.context, top_k= 1)[0]["score"] > 0.5:

               res[q] = layout_model(question=q, **doc.context, top_k= 1)[0]["answer"]
               
    if pred == "addendum":
         for q in ["stamp duty?","certificate number","issued date", "issued by?","state?","puchased by?","stamp duty paid by?","first party?","second party?"]:
              print(q, p(question=q, **doc.context, top_k= 1)[0]["answer"])
              
    if pred == "birth certificate":
         for q in ["name of child?","mother name","sex", "date of birth?","place of birth?","name of father?","registration number","registration date"]:
              print(q, p(question=q, **doc.context, top_k= 1)[0]["answer"])
              
               
    if pred == "Bank Statement":
        kws = ["customer id?","branch name","IFSC code?", "micr code?","account number?","statement period?","customer name",\
                     "date","account type","currency","account branch","cost ID","IFS code"]
            
        for q in kws:
            if layout_model(question=q, **doc.context, top_k= 1)[0]["score"] > 0.5:

               res[q] = layout_model(question=q, **doc.context, top_k= 1)
                   
                   
    if pred == "CA CS Certificate for Sec 180 1 ( c )":
        kws = ["credit arrangement letter number","AGM date"]
        for q in kws:
            if layout_model(question=q, **doc.context, top_k= 1)[0]["score"] > 0.5:

               res[q] = layout_model(question=q, **doc.context, top_k= 1)[0]["answer"]
              
    if pred == "Director'sBorrower's Undertaking":
        kws = ["stamp duty?","certificate number","issued date", "issued by?","state?","puchased by?","stamp duty paid by?",\
               "first party?","second party?","loan amount", "borrower name", "CAL number"]
        for q in kws:
            if layout_model(question=q, **doc.context, top_k= 1)[0]["score"] > 0.5:
               res[q] = layout_model(question=q, **doc.context, top_k= 1)[0]["answer"]
               
    if pred == "Salary Slip":
                 results = model(im, size=640)
                 final_res = pd.DataFrame()
                 p = results.pandas().xyxy[0]
                 
                 for idx, (name,x1,y1,x2,y2,conf) in enumerate(zip(p["name"],p["xmin"],p["ymin"],p["xmax"],p["ymax"],p["confidence"])):
                     x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                     cv2.rectangle(predicted, (x1, y1), (x2, y2), (255,0,0), 3)
                     cv2.putText(predicted,str(name)+ "-->" + str(round(conf, 2)), (x1,y1-30), 0,2, (0,255,0),5) 
                     crop_img = inp.copy()[y1:y2, x1:x2]
                     crop_path = "tempDir/"+"crop"+str(idx)+".jpg"
                     cv2.imwrite(crop_path, crop_img)
                     words_loc_actual = apply_ocr(crop_path)
                     df1_transposed = get_ocr(crop_img, words_loc_actual)
                     final_res = pd.concat([final_res, df1_transposed], axis=0)
                 res = final_res.to_json()
                 print(res)

    print(res)
    return (pred, res, predicted)

gr.Interface(fn=predict, 
#             inputs=gr.Image(type="pil"),
             inputs=gr.Image(),
             outputs=[gr.Label(num_top_classes=3), gr.JSON(), gr.Image()],

        #     outputs=gr.Label(num_top_classes=3),
             examples=[]).launch(share=False)