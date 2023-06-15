from doctr.io import DocumentFile
from doctr.models import ocr_predictor
model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
import itertools
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    # PDF
    # doc = DocumentFile.from_pdf("RTL FA for high rated Borrowers_Wonder Cement.pdf")
    doc = DocumentFile.from_images("temp_file.jpg")

    # Analyze
    result = model(doc)

#    result = model(doc[5:6])
    json_output = result.export()
    words_loc_normalized = []
    corpus = []
    for ii in range (len(json_output['pages'])):
      for d in json_output.values():
          num_blocks = 6#len(d[ii]['blocks'])
      #    print(num_blocks)
  
          for k in range (num_blocks):
          
            row = d[ii]['blocks'][k]['lines']
            print(row)
       #     res = " ".join([k['value'] for item in row for k in item['words']])
         #   print(res)
        #    corpus.append(res)
 #   print(corpus)
    '''      
          for i in range(num_blocks):
            num_lines = len(d[0]['blocks'][i]['lines']) 
          #  print(i, num_blocks[i])
            
            for j in range(num_lines):
                row = ' '.join([word['value'] for word in d[0]['blocks'][i]['lines'][j]['words']])
                print(row)
                
          #      myfile.write("%s\n" % row)
       
                words_loc_normalized.append([(word['value'],word['geometry']) for word in d[0]['blocks'][i]['lines'][j]['words']])
          #  st.write(row)
      # flatten list of lists
    words_loc_normalized = list(itertools.chain(*words_loc_normalized))
      
      # convert relative pixel coordinates to actual image coordinates
    h, w = (json_output['pages'][0]['dimensions']); #print(w,h)
    words_loc_actual = list(map(lambda x: (x[0],(int(x[1][0][0]*w),int(x[1][0][1]*h)),(int(x[1][1][0]*w),int(x[1][1][1]*h))) , words_loc_normalized))
      
    print(words_loc_actual)
    '''