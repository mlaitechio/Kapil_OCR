# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 22:05:12 2023

@author: kapil
"""


from haystack import Pipeline, Document
from haystack.utils import print_answers
from haystack.nodes import FARMReader

new_reader = FARMReader(model_name_or_path="br_model")

p = Pipeline()
p.add_node(component=new_reader, name="Reader", inputs=["Query"])
res = p.run(
    query="what is the largest city in karnataka? ", documents=[Document(content=context)]
)
print_answers(res,details="medium")