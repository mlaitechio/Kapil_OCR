# Imports
import base64
import PyPDF2
from PyPDF2 import PdfFileMerger
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from utils import HocrParser
import re
import fitz

def find_clause_in_doc(pattern, file_name):
# search specific words in the pdf and print all matches
#pattern = "IN THE FACILITY AGREEMENT"
#file_name = "hocr_output.pdf"
    '''
    reader = PyPDF2.PdfFileReader(file_name)
    num_pages = reader.getNumPages()
    
    for i in range(0, num_pages):
        page = reader.getPage(i)
        text = page.extractText()
        
        for match in re.finditer(pattern, text):
            print(f'Page no: {i} | Match: {match}')
            
    '''        
    
    # load document
    doc = fitz.open(file_name)
    
    # get text, search for string and print count on page.
    for page in doc:
        text = ''
        text += page.get_text()
        text = ''.join(text.splitlines())
        print(f'count on page {page.number +1} is: {len(re.findall(pattern, text))}')
        for match in re.finditer(pattern, text):
            print(f'Page no: {page.number +1} | Match: {match}')
            print(match.group())
      #      print(text)
      #      preprocess = re.split('\s{4,}',text)
      #      print(preprocess)

            
        

def highlight_text_in_doc(text, file):
    ### READ IN PDF
    doc = fitz.open(file)
    
    for page in doc:
        ### SEARCH
     #   print(page.get_text("words"))
     #   text = "In the Facility Agreement, unless there is anything repugnant to the subject or context"
        text_instances = page.search_for(text)
     #   print(text)
        ### HIGHLIGHT
        for inst in text_instances:
            highlight = page.add_highlight_annot(inst)
            highlight.update()
    '''        
    wordlist = page.get_text("words")
    wordlist.sort(key=lambda w: (w[1], w[0]))  # sort vertical, then horizontal
    for w in wordlist:
        if w[4] == "Additional":  # choose a word to start marking
            pointa = fitz.Point(w[:2])  # top left of work rectangle
            break
    for w in wordlist:
        if w[4] == "Agreement.":  # some word for stopping the marking
            pointb = fitz.Point(w[2:4])  # bottom right of word rectangle
            break
    
    page.add_highlight_annot(start=pointa, stop=pointb)  # a less well known form of hightlight annotations
    '''
    
    ### OUTPUT
    doc.save("highlighted_output.pdf", garbage=4, deflate=True, clean=True)
    

if __name__ == '__main__':
    
    docs = DocumentFile.from_pdf("RTL FA for high rated Borrowers_Wonder Cement.pdf")
    model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
    # we will grab only the first two pages from the pdf for demonstration
    result = model(docs[:5])
    result.show(docs)
    '''

    # returns: list of tuple where the first element is the (bytes) xml string and the second is the ElementTree
    xml_outputs = result.export_as_xml()

    # init the above parser
    parser = HocrParser()
    
    # export xml file
    xml_outputs = result.export_as_xml()
    with open("doctr_image_hocr.xml","w") as f :
        f.write(xml_outputs[3][0].decode())
        
        
    from ocrmypdf.hocrtransform import HocrTransform
    output_pdf_path = "hocr_output.pdf"

    hocr = HocrTransform(
        hocr_filename="doctr_image_hocr.xml",
        dpi=300
    )

    # step to obtain ocirized pdf
    hocr.to_pdf(
        out_filename=output_pdf_path,
        image_filename=None,
        show_bounding_boxes=False,
        interword_spaces=True,
    )
    '''
#    find_clause_in_doc("In the Facility Agreement, unless there is anything repugnant to the subject or context thereof, the expressions listed below shall have the following meaning: ", "RTL FA for High Rated Borrower_Sample 1.pdf")
    
  #  highlight_text_in_doc("In the Facility Agreement, unless there is anything repugnant to the subject or context thereof, the expressions listed below shall have the following meaning: ", "hocr_output.pdf")
  #  highlight_text_in_doc("thereof, the expressions listed below shall have the following meaning:", "hocr_output.pdf")
    