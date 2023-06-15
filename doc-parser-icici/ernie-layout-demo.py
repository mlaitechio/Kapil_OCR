# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 16:52:18 2023

@author: kapil
"""

from pprint import pprint
from paddlenlp import Taskflow
import json


with open('config.json', 'r') as f:
  config = json.load(f)

print(config)

docprompt = Taskflow("document_intelligence")
pprint(docprompt([{"doc": r"top-20\Board Resolution\123602012020.TIF", "prompt": config['Board Resolution']}]))



#pprint(docprompt([{"doc": r"D:\MLAI-2\doc-parser-icici\icici 547 docs\Revival letter\320558507022017.jpg", "prompt": config['Revival letter']}]))

#pprint(docprompt([{"doc": "icici 547 docs\Bank Statement/LBABD00006241182_Bank Statement-0 - Copy.jpg", "prompt": config["Bank Statement"]}]))
'''    
pprint(docprompt([{"doc": "icici 547 docs\addendum/7723702989-2.jpg", "prompt": ["stamp duty?","certificate number","issued date", "issued by?","state?",
                "puchased by?","stamp duty paid by?","first party?","second party?"]}]))

pprint(docprompt([{"doc": "icici 547 docs\Director'sBorrower's Undertaking/16699321122020-0.jpg", "prompt": ["stamp duty?","certificate number","issued date", 
                    "issued by?","state?","puchased by?","stamp duty paid by?","first party?","second party?","loan amount", "borrower name", "CAL number"]}]))


pprint(docprompt([{"doc": "icici 547 docs\Accepted CAL/26370008122020-2.jpg", "prompt": ["stamp duty?","certificate number","issued date", 
                      "issued by?","state?","puchased by?","stamp duty paid by?","first party?","second party?","loan amount", "borrower name", "CAL number"]}]))

# schema = ['时间', '选手', '赛事名称'] # Define the schema for entity extraction
# ie = Taskflow('information_extraction', schema=schema, model='uie-base')
# ie_en = Taskflow('information_extraction', schema=schema, model='uie-base-en')

'''
#similarity = Taskflow("text_similarity")
#cc = similarity([["“IBC” means the Insolvency and Bankruptcy Code, 2016, including all replacements and amendments made thereto and all rules and regulations framed thereunder. ", "“IBC” means the Insolvency and Bankruptcy Code, 2016, excluding all replacements and amendments made thereto and all rules and regulations framed thereunder. "]])