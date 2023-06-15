from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from pprint import pprint

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
model_ocr = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
import itertools
import warnings
import gradio as gr
warnings.filterwarnings("ignore")

def find_ocr(file):
    print(file.name)
    doc = DocumentFile.from_pdf(str(file.name))

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
      #      print(row)
            res = " ".join([k['value'] for item in row for k in item['words']])
         #   print(res)
            corpus.append(res)
            
    return " ".join(corpus)


def extract_values(text, docs):
    
    results = {question:query_from_list(question,  text, 30)  for question in prompts[str(docs)]}
    return results
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(device)

def query_from_list(query, options, tok_len):
    t5query = f"""Question: "{query}" Context: {options}"""
    inputs = tokenizer(t5query, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=tok_len)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

# specify you question here. 
prompts = {"gst_invoices": ["bank name","branch address", "invoice number"]} 

#%%time
if __name__ == "__main__":
    
    # Call the LLM with input data and instruction
 #   raw_text = find_ocr("top-20/Board Resolution/3023002012020.PDF")
    
 #   input_data= " ".join(raw_text)
 #   results = {question:query_from_list(question,  input_data, 30)  for question in prompts["ca_certified_networth"]}
 #   pprint(results)
     
 
    with gr.Blocks() as demo:
    
         file = gr.File()
         docs = gr.Radio(["corporate_gurantee","gst_invoices"], label="Doc_type")
         print(docs)
    
         output = gr.Textbox()
         btn = gr.Button(value="Get OCR")
         btn.click(find_ocr, inputs=[file], outputs=[output])
         
         with gr.Row():
             output2 = gr.JSON()
    
         btn = gr.Button(value="Extract info")
         btn.click(extract_values, inputs=[output, docs], outputs=[output2])
         
    demo.launch()