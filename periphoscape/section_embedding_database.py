import os
import sys
import gzip
import pickle
import numpy as np

from periphoscape.constants import _T
from periphoscape.util import l2_distance, cos_similarity, normalize_matrix, matrix_similarity


class SectionEmbeddingDB():
    
    def __init__(self, page_db, search_agent, always_vector_cache=False):
        self.page_db = page_db
        self.search_agent = search_agent
        self.db = {}
        self.page_vector_cache = {}
        #
        self.get_page_vector_function = self.get_page_vector
        self.get_cached_page_vector_function = self.get_cached_page_vector

    
    def load_page_vector_cache(self, filename):
        with gzip.open(filename, 'rb') as f:
            self.page_vector_cache = pickle.load(f)

            
    def load_db(self, filename):
        with gzip.open(filename, 'rb') as f:
            self.db = pickle.load(f)

            
    def save_db(self, filename):
        with gzip.open(filename, 'wb') as f:
            pickle.dump(self.db, f)

        
    def fetch_data(self, page_id, link_size_limit=100):
        r = self.search_agent.get_search_results(str(page_id), size=link_size_limit, 
                                                source=['data_type', 'vector'], fields=['wiki_id'])
        self.store_data(page_id, r)
    
    
    def clear(self):
        self.db = {}
        
        
    def store_data(self, page_id, embeddg_list):
        page_matrix = []
        section_matrix = []
        for embedding in embeddg_list:
            if embedding['data_type'] == 'page':
                page_matrix.append(embedding['vector'])
            elif embedding['data_type'] == 'section':
                 section_matrix.append(embedding['vector'])
        self.db[page_id] = { 'page'    : np.array(page_matrix),
                             'section' : np.array(section_matrix) }

        
    def has_entry(self, page_id):
        return page_id in self.db

    
    def is_not_ailable(self, page_id):
        return (page_id in self.db) and (self.db.get(page_id) is None)
    
    
    def get_page_matrix(self, q):
        page_id, page_name = self.page_db.get_id_and_name(q)
        if page_id is None:
            return None
        return self.get_page_matrix_by_id(page_id)


    def get_page_matrix_by_id(self, page_id):
        if not self.has_entry(page_id):
            self.fetch_data(page_id)
        d = self.db.get(page_id)
        if d is None:
            return None
        return d['page']

    
    def get_section_matrix(self, q):
        page_id, page_name = self.page_db.get_id_and_name(q)
        if page_id is None:
            return None
        return self.get_section_matrix_by_id(page_id)
    
    
    def get_section_matrix_by_id(self, page_id):
        if not self.has_entry(page_id):
            self.fetch_data(page_id)
        d = self.db.get(page_id)
        if d is None:
            return None
        return d['section']

    
    def get_page_vector(self, q):
        m = self.get_page_matrix(q)
        if m is None or len(m) == 0:
            return None
        return m[0]
    

    def get_cached_page_vector(self, q):
        page_id, page_name = self.page_db.get_id_and_name(q)
        if page_id is None:
            return None
        return self.page_vector_cache.get(page_id)

    
    def enable_page_vector_cache(self):
        self.get_cached_page_vector = self.get_cached_page_vector_function

        
    def disable_page_vector_cache(self):
        self.get_cached_page_vector = self.get_page_vector_function
    
    
    def get_category_vector(self, cat_id):
        l = self.page_db.lookup_links('category', _T.CATEGORY, _T.ARTICLE, _T.BACKWARD, cat_id,
                                      None)
        if l is None:
            return None
        vl = []
        for page_id in l:
            v = self.get_cached_page_vector(page_id)
            if v is not None:
                vl.append(v)
            if len(vl) == 0:
                return None
        return np.average(vl, axis=0)

    
    def get_category_vector_x(self, cat_id):
        pages = []
        for db in ['category', 'page']:
            l = self.page_db.lookup_links(db, _T.CATEGORY, _T.ARTICLE, _T.BACKWARD, cat_id, [])
            pages.extend(l)
        if len(pages) == 0:
            return None
        vl = []
        for page_id in pages:
            v = self.get_cached_page_vector(page_id)
            if v is not None:
                vl.append(v)
            if len(vl) == 0:
                return None
        return np.average(vl, axis=0)

    
    # pair = ( (type1, id1), (type2, id2) )
    #
    def get_similarity(self, pair):
        vectors = [None, None]
        for i in [0, 1]:
            element_type, element_id = pair[i]
            if element_type == _T.ARTICLE:
                vectors[i] = self.get_cached_page_vector(element_id)
            else:
                vectors[i] = self.get_category_vector(element_id)
        if vectors[0] is None or vectors[1] is None:
            return None
        return cos_similarity(vectors[0], vectors[1])

    
    def get_distance(self, pair):
        vectors = [None, None]
        for i in [0, 1]:
            element_type, element_id = pair[i]
            if element_type == _T.ARTICLE:
                vectors[i] = self.get_cached_page_vector(element_id)
            else:
                vectors[i] = self.get_category_vector(element_id)
        if vectors[0] is None or vectors[1] is None:
            return None
        return l2_distance(vectors[0], vectors[1])

    
    # get section-base similarity of two articles.
    #
    def get_section_base_similarity(self, q1, q2):
        m1 = self.get_section_matrix(q1)
        if m1 is None or len(m1) == 0:
            return None
        m2 = self.get_section_matrix(q2)
        if m2 is None or len(m2) == 0:
            return None
        return matrix_similarity(m1, m2)

    
    def get_page_base_similarity(self, q1, q2):
        v1 = self.get_page_vector(q1)
        if v1 is None or len(v1) == 0:
            return None
        v2 = self.get_page_vector(q2)
        if v2 is None or len(v2) == 0:
            return None
        v1 = normalize_matrix(v1, axis=0)
        v2 = normalize_matrix(v2, axis=0)
        return (v1 @ v2.T)

    
    def get_page_section_similarity(self, *, page_base, section_base):
        mp = self.get_page_matrix(page_base)
        if mp is None:
            return None
        ms = self.get_section_matrix(section_base)
        if ms is None:
            return None
        return matrix_similarity(mp, ms)

    
    # get similarities between q and its adjacent elements
    #
    def get_similarity_of_links(self, q, direction=_T.FORWARD):
        page_id, page_name = self.page_db.get_id_and_name(q)
        if page_id is None:
            return None
        page_matrix_0 = self.get_page_matrix(page_id)
        section_matrix_0 = self.get_section_matrix(page_id)
        if page_matrix_0 is None or section_matrix_0 is None:
            return None
        #
        links = self.page_db.lookup_links('page', _T.ARTICLE, _T.ARTICLE, direction, page_id, None)
        if links is None:
            return None
        page_matrix_list = [ self.get_page_matrix(l) for l in links ]
        section_matrix_list = [ self.get_section_matrix(l) for l in links ]
        #
        r = []
        for i in range(len(links)):
            d_name = self.page_db.get_name_by_id(links[i])
            #
            m1 = page_matrix_list[i]
            if m1 is None or len(m1) == 0:
                s_p_p = None
                s_s_p = None
            else:
                s_p_p = matrix_similarity(page_matrix_0, m1)
                s_s_p = matrix_similarity(section_matrix_0, m1)
            #
            m2 = section_matrix_list[i]
            if m2 is None or len(m2) == 0:
                s_p_s = None
                s_s_s = None
            else:
                s_p_s = matrix_similarity(page_matrix_0, m2)
                s_s_s = matrix_similarity(section_matrix_0, m2)
            r.append( (d_name, s_p_p, s_s_s, s_p_s, s_s_p) )
        return r
