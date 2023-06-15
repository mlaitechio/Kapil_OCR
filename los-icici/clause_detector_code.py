import torch
import numpy as np
import cv2
from sentence_transformers import SentenceTransformer, util

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
model_doctr = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
import itertools
import warnings
warnings.filterwarnings("ignore")

def pointInRect(out,rect):
    res = []
    for t,px,py in out:
        x1,y1 = rect[0],rect[1]
        x2,y2 = rect[2],rect[3]
        if (x1 < px and px < x2):
            if (y1 < py and py < y2):
               res.append(t)   
    return " ".join(res)

def ocr_get(file):
 #   doc = DocumentFile.from_pdf(file)
    # Analyze
 #   result = model_doctr(doc[3:4])
    doc = DocumentFile.from_images(file)

    result = model_doctr(doc)
    json_output = result.export()
    words_loc_normalized = []
  #  corpus = []
    for ii in range (len(json_output['pages'])):

      for d in json_output.values():
          num_blocks = len(d[ii]['blocks'])
       
          for i in range(num_blocks):
            num_lines = len(d[ii]['blocks'][i]['lines']) 
          #  print(i, num_blocks[i])
            
            for j in range(num_lines):
          #      row = ' '.join([word['value'] for word in d[0]['blocks'][i]['lines'][j]['words']])
           #     print(row)
                       
                words_loc_normalized.append([(word['value'],word['geometry']) for word in d[ii]['blocks'][i]['lines'][j]['words']])
          #  st.write(row)
      # flatten list of lists
    words_loc_normalized = list(itertools.chain(*words_loc_normalized))
      
      # convert relative pixel coordinates to actual image coordinates
    h, w = (json_output['pages'][0]['dimensions']); #print(w,h)
    words_loc_actual = list(map(lambda x: (x[0],(int(x[1][0][0]*w),int(x[1][0][1]*h)),(int(x[1][1][0]*w),int(x[1][1][1]*h))) , words_loc_normalized))
      
 #   print(words_loc_actual)
    
    raw_ocr = [(t,int((x1+x2)/2),int((y1+y2)/2)) for (t,(x1,y1),(x2,y2)) in words_loc_actual]
    return raw_ocr

if __name__ == "__main__":

    raw_ocr = ocr_get("extracted-images/img34.jpg")
    corpus = []
    
    model = torch.hub.load('ultralytics/yolov5', 'custom', path= "clause_detector.pt", force_reload=False)  # local model inv_250.pt
    predicted = cv2.imread("extracted-images/img32.jpg")  
    orig = predicted.copy()
    model.conf = 0.6
    results = model("extracted-images/img34.jpg", size=640)
      
    p = results.pandas().xyxy[0]
     
    for name,x1,y1,x2,y2,conf in zip(p["name"],p["xmin"],p["ymin"],p["xmax"],p["ymax"],p["confidence"]):
         x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
         cv2.rectangle(predicted, (x1, y1), (x2, y2), (0,0,255), 5)
    #         cv2.putText(predicted,str(name)+ "-->" + str(round(conf, 2)), (x1,y1-30), 0,2, (0,255,0),5) 
         cv2.putText(predicted,str(name), (x1,y1-30), 0,2, (0,255,0),5)  
         crop_img = orig[int(y1):int(y2), int(x1):int(x2)]
       #  cv2.imwrite("temp_file.jpg", crop_img)
         rect = x1,y1,x2,y2
         rect_ocr = pointInRect(raw_ocr, rect)  
    #     print(rect_ocr)
    #     print("=====")
         corpus.append(rect_ocr)
         
    cv2.imshow("out", predicted)
    cv2.waitKey(0)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
       # Query sentences:
    queries = ["“Credit Arrangement Letter” or “CAL” means a letter, as of the date specified in the Schedule I, issued by ICICI Bank to the Borrower, granting the Facility to the Borrower. The expression CAL shall include all amendments to the CAL."]#list(itertools.chain(*out))[:2]
       
       # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = min(1, len(corpus))
    for query in queries:
           query_embedding = embedder.encode(query, convert_to_tensor=True)
       
           # We use cosine-similarity and torch.topk to find the highest 5 scores
           cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
           top_results = torch.topk(cos_scores, k=top_k)
       
           print("\n\n======== pdf clause det ==============\n\n")
           print("Query:", query)
           print("most similar sentences in corpus:")
       
           for score, idx in zip(top_results[0], top_results[1]):
               print(corpus[idx], "(Score: {:.4f})".format(score))
             #  return corpus[idx]
    
