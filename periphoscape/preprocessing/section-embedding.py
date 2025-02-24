import numpy as np
import json
from sentence_transformers import SentenceTransformer

def embed_sections(page_data, model, query_format, delimiter='。'):
    title = page_data.get('title')
    if title is None:
        return None
    embeddings = { 'title' : title, 'section' : {} }
    all_sentence_vectors = []
    section_vectors = []    
    for section_title, section_contents in page_data['section'].items():
        sentences = [ query_format(s) for s in ''.join(section_contents).split(delimiter) if s != '' ]
        if sentences:
            sentence_vectors = model.encode(sentences)
            section_vector =  np.average(sentence_vectors, axis=0)
            all_sentence_vectors.extend(sentence_vectors)
            embeddings['section'][section_title] = section_vector.tolist()
            section_vectors.append(section_vector)
        else:
            embeddings['section'][section_title] = None
    if section_vectors:
        page_vector = np.average(all_sentence_vectors, axis=0)
        embeddings['page'] = page_vector.tolist()
    else:
        embeddings['page'] = None
    return embeddings


# Examples of model names and their query formats:
#
#   model_name = "intfloat/multilingual-e5-large"
#   query_format = lambda x: f'query: {x}'
#
#   model_name = "cl-nagoya/sup-simcse-ja-base"
#   uery_format = lambda x: x
#
def section_jsonl2embedding_jsonl(model_name, query_format,
                                  section_jsonl_filename, embedding_jsonl_filename):
    model = SentenceTransformer(model_name, device='cuda')
    with open(embedding_jsonl_filename, 'w') as fout:
        with open(section_jsonl_filename, 'r') as fin:
            while True:
                l = fin.readline()
                if not l:
                    break
                j = json.loads(l)
                embds = embed_sections(j, model, query_format)
                print(json.dumps(embds, ensure_ascii=False), file=fout)

