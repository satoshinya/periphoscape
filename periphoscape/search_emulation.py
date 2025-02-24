import gzip
import json
import numpy as np

#from periphoscape.dataset import Dataset

class ElasticSearchEmulation():

    def __init__(self, page_db, dataset, embedding_dimension=768):
        self.page_db = page_db
        self.load_search_results(dataset.files['search_results'])
        self.rng = np.random.default_rng()
        self.embedding_dimension = embedding_dimension
        self.miss = set()

        
    def load_search_results(self, cache_filename):
        with gzip.open(cache_filename, 'rb') as f:
            print('Loading...', end='')
            self.cache = json.load(f)
            print('done.')

        
    def make_multi_match_query(self, must_terms, should_terms, **args):
        return (must_terms, args)

    
    def do_search(self, q):
        query = q[0][0]
        #
        m = self.cache.get(query)
        if m is None:
            id_ = int(query)
            title = self.page_db.get_name_by_id(id_)
            m = self.generate_random_response(title, id_)
            self.cache[query] = m
            #
            self.miss.add(id_)
            #
            print(f'** [ElasticSearchEmulation] missing "{title}" ({id_})')
        return m
        

    def generate_random_response(self, title, id_):
        v = self.rng.random(self.embedding_dimension)
        r = [
            { 
                '_source' : 
                {
                    'data_type' : 'page', 
                    'header' : title, 
                    'vector'  : v, 
                    'title' : title, 
                    'wiki_id' : id_ 
                }         
            },
            { 
                '_source' : 
                {
                    'data_type' : 'section', 
                    'header' : '_Lead', 
                    'vector'  : v, 
                    'title' : title, 
                    'wiki_id' : id_ 
                }
            }
        ]
        return {'hits' : {'hits' : r }}
