import fitz

### READ IN PDF
doc = fitz.open("hocr_output (5).pdf")

for page in doc:
    ### SEARCH
    text = "In the Facility Agreement, unless there is anything repugnant to the subject or context"
    text_instances = page.search_for(text)

    ### HIGHLIGHT
    for inst in text_instances:
        highlight = page.add_highlight_annot(inst)
        highlight.update()


### OUTPUT
doc.save("output.pdf", garbage=4, deflate=True, clean=True)