import gradio as gr
import pandas as pd
import itertools
import textract
import cv2
import torch
from sentence_transformers import SentenceTransformer, util
import fitz

import docx2txt

from difflib import Differ

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
model_doctr = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)


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

def extra(file, query_clause):
    raw_ocr = ocr_get(file.name)
    corpus = []
    
    model = torch.hub.load('ultralytics/yolov5', 'custom', path= "clause_detector.pt", force_reload=False)  # local model inv_250.pt
    predicted = cv2.imread(file.name)  
    orig = predicted.copy()
    model.conf = 0.6
    results = model(file.name, size=640)
      
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
         
 #   cv2.imshow("out", predicted)
 #   cv2.waitKey(0)
#    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
       # Query sentences:
    queries = [query_clause]
       
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
               return corpus[idx], "(Score: {:.4f})".format(score)
    


def highlight_text_in_doc(text, file):
    ### READ IN PDF
    doc = fitz.open(file)
    
    for page in doc:
        ### SEARCH
        print(page.get_text("words"))
     #   text = "In the Facility Agreement, unless there is anything repugnant to the subject or context"
        text_instances = page.search_for(text)
        print(text)
        ### HIGHLIGHT
        for inst in text_instances:
            highlight = page.add_highlight_annot(inst)
            highlight.update()
    doc.save("highlighted_output.pdf", garbage=4, deflate=True, clean=True)

def read_file(file):
    print(file.name)
    text = docx2txt.process(file.name)

    
    content = []
    for line in text.splitlines():
      #This will ignore empty/blank lines. 
      if line != '':
        #Append to list
        content.append(line)
    
 #   print (content)
    return content

def copmare_clause_in_docs(file, query_clause):

    corpus = read_file(file)
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    # Query sentences:
        
 #   df = pd.read_excel("High Rated RTL FA_Standard Clause library_v6.xlsx", usecols='F:F')
 #   out = df.dropna()
 #   out = out.values.tolist()
    queries = [query_clause]#list(itertools.chain(*out))[:2]
    
    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = min(1, len(corpus))
    for query in queries:
        query_embedding = embedder.encode(query, convert_to_tensor=True)
    
        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)
    
        print("\n\n======================\n\n")
        print("Query:", query)
        print("\nTop 2 most similar sentences in corpus:")
    
        for score, idx in zip(top_results[0], top_results[1]):
            print(corpus[idx], "(Score: {:.4f})".format(score))
            return corpus[idx]
        
        
def predict(inp, row):
    df = pd.read_excel("High Rated RTL FA_Standard Clause library_v6.xlsx", usecols='F:F')
    out = df.dropna()
 #   out_values = out.values.tolist()
  #  queries = list(itertools.chain(*out))[:10]
  #  print(queries)
#    print(row, type(row))
    print(out.iloc[int(row)][0])
    return out, out.iloc[int(row)][0]
    
def diff_texts(text1, text2):
    d = Differ()
    return [
        (token[2:], token[0] if token[0] != " " else None)
        for token in d.compare(text1, text2)
    ]
'''
gr.Interface(fn=predict, 
             inputs=[gr.File(),"number",gr.Button("Process")],
  #           inputs= [gr.Image(source="webcam", shape = (600, 400), streaming = True)],  # source = "webcam"
             outputs=[gr.Dataframe(),"text"],

             capture_session = True, 
             title="Cylinder head inspection example",
             examples=[]).launch(share=False)  #auth=("admin", "tata1234")
'''

with gr.Blocks() as demo:

    file = gr.File()
    file2 = gr.Number()

    output = gr.Dataframe(label = "Clause list",row_count=5, max_rows = 5, overflow_row_behaviour = "pagination")
    output2 = gr.Text()
    btn = gr.Button(value="Submit")
    btn.click(predict, inputs=[file, file2], outputs=[output, output2])
    
    with gr.Row():
        file = gr.File(file_types = [".docx"])
        im_2 = gr.Text()

    btn = gr.Button(value="Extract clause")
    btn.click(copmare_clause_in_docs, inputs=[file, output2], outputs=[im_2])
    
    with gr.Row():
        file = gr.File()
        
    btn = gr.Button(value="highlight pdf")

    btn.click(highlight_text_in_doc, inputs=[im_2, file], outputs=[im_2])
    
    with gr.Row():
        piss = gr.HighlightedText(label="Diff",combine_adjacent=True,show_legend=True,).style(color_map={"+": "red", "-": "green"})
    #    theme=gr.themes.Base()

    btn = gr.Button(value="highlight pdf using difference")

    btn.click(diff_texts, inputs=[output2, im_2], outputs=[piss])
    
    with gr.Row():
        file = gr.File()
        im_3 = gr.Text()

    btn = gr.Button(value="clause in pdf")

    btn.click(extra, inputs=[file, output2], outputs=[im_3])
    

if __name__ == "__main__":
     embedder = SentenceTransformer('all-MiniLM-L6-v2')
   #  get_ocr("RTL FA for high rated Borrowers_Wonder Cement.pdf")

     demo.launch()