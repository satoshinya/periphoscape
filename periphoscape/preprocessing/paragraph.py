from mwparserfromhtml import HTMLDump
import pprint
import json

def html2section_jsonl(html_filename, section_jsonl_filename):
    html_dump = HTMLDump(html_filename)
    fout = open(section_jsonl_filename, 'w')
    for article in html_dump:
        jl = { 'title' : article.get_title(), 'section' : {} }
        prev_heading = "_Lead"
        jl['section'][prev_heading] = []
        for heading, paragraph in article.html.wikistew.get_plaintext(exclude_transcluded_paragraphs=True,
                                                                      exclude_para_context=None,  
                                                                      exclude_elements={"Heading",
                                                                                        "Math",
                                                                                        "Citation",
                                                                                        "List",
                                                                                        "Wikitable",
                                                                                        "Reference"}):
            if heading != prev_heading:
                jl['section'][heading] = []
                prev_heading = heading
            jl['section'][heading].append(paragraph.replace('\u3000', ' '))
        print(json.dumps(jl, ensure_ascii=False), file=fout)
    fout.close()
