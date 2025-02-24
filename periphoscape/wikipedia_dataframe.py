import pandas as pd
import numpy as np
import os
import sys

from periphoscape.constants import _T
from periphoscape.util import cos_similarity, get_average_similarity, matrix_similarity
from periphoscape.wikipedia_network import *


def make_neighborhood_dataframe(q, page_db, wiki2vec, section_embedding):
    q_id = page_db.get_id_by_name(q)
    if q_id is None:
        return None
    wiki_df = WikipediaDataFrame(page_db, wiki2vec, section_embedding)
    wiki_df.add_neighborhood_columns(q_id)
    wiki_df.retain_data('redirect', 0)
    wiki_df.add_ego_similarity_columns()
    return wiki_df


class WikipediaDataFrame():
    def __init__(self, page_db, wikipedia2vec, section_embedding, df=None):
        self.page_db = page_db
        self.wikipedia2vec = wikipedia2vec
        self.section_embedding = section_embedding
        if df is None:
            self.df = pd.DataFrame()
        else:
            self.df = df

    def copy(self, df=None):
        return WikipediaDataFrame(self.page_db, self.wikipedia2vec, self.section_embedding, df=df)
        
    # select samples,  and make dfs[label] for the samples
    #
    def select_page_samples(self, num_samples):
        rng = np.random.default_rng()
        sample_ids = set()
        max_page_id = self.page_db.get_max_page_id()
        i = 0
        while True:
            j = rng.integers(max_page_id, size=1)[0]
            if j in self.page_db.id2name[_T.ARTICLE]:
                if j not in sample_ids:
                    sample_ids.add(j)
                    i += 1
                    if i == num_samples:
                        break         
        #
        return self.add_page_columns(sorted(sample_ids))
   

    def add_page_columns(self, id_list):
        self.df['id'] = id_list
        self.df['type'] = _T.ARTICLE
        self.df['name'] = [ self.page_db.lookup_id2name(_T.ARTICLE, i) for i in id_list ]
        self.df['redirect'] = [ self.page_db.lookup_id2redirect(_T.ARTICLE, i) for i in id_list ]
        return self


    def get_id_list(self, id_column='id'):
        return [ int(i) for i in self.df[id_column].values ]
    
    def add_neighborhood_columns(self, page_id, outgoing_column='out_flag', incoming_column='in_flag'):
        neighbors = {}
        for direction in [_T.FORWARD, _T.BACKWARD]:
            for neighbor in self.page_db.get_adjacent_elements(page_id, _T.ARTICLE, directions=[direction],
                                                        adjacent_element_types=[_T.ARTICLE]):
                neighbor_id = neighbor[1]
                if neighbor_id not in neighbors:
                    neighbors[neighbor_id] = []
                neighbors[neighbor_id].append(direction)                
        id_list = list(neighbors)
        # neighbors may contain page_id if there is a self-loop
        if page_id not in id_list:
            id_list.append(page_id)
        self.add_page_columns(id_list)
        #
        # links to page_id (self-loop) is removed
        self.df[outgoing_column] = [ 1 if ((_T.FORWARD in neighbors.get(i, [])) and (i != page_id)) else 0
                                     for i in id_list ]
        self.df[incoming_column] = [ 1 if ((_T.BACKWARD in neighbors.get(i, [])) and (i != page_id))  else 0
                                     for i in id_list ]


    
    def get_egoid_in_neighborhood_dataframe(self, 
                                            outgoing_flag_column='out_flag', incoming_flag_column='in_flag'):
        return int(self.df[self.df[outgoing_flag_column] + self.df[incoming_flag_column]==0]['id'].values[0])

    
    def get_nonego_elements_in_neighborhood_dataframe(self, 
                                            outgoing_flag_column='out_flag', incoming_flag_column='in_flag'):
        return self.df[self.df[outgoing_flag_column] + self.df[incoming_flag_column] > 0]['id'].values

        
    def add_ego_similarity_columns(self, 
                                    page_base_column='ego_sim_page', 
                                    section_base_column='ego_sim_section',
                                    outgoing_flag_column='out_flag', incoming_flag_column='in_flag'):
        ego_id = self.get_egoid_in_neighborhood_dataframe(outgoing_flag_column=outgoing_flag_column,
                                                          incoming_flag_column=incoming_flag_column)
        #
        ego_embedding = self.get_section_page_vector(ego_id)
        if ego_embedding is None:
            print(f'Cannot get embedding of {ego_id}')
            self.df[page_base_column] = None
        ego_matrix = self.get_section_matrix_by_id(ego_id)
        if ego_matrix is None:
            print(f'Cannot get section matrix of {ego_id}')
            self.df[section_base_column] = None
        else:
            self.df[page_base_column] = [ cos_similarity(ego_embedding, self.get_section_page_vector(i)) 
                                           for i in self.df['id'] ]
            self.df[section_base_column] = [ matrix_similarity(ego_matrix, self.get_section_matrix_by_id(i))
                                            for i in self.df['id'] ]
    

    def make_network_data_from_neighborhood_dataframe(self, include_ego=True,
                                                      id_column='id',
                                                      outgoing_flag_column='out_flag', 
                                                      incoming_flag_column='in_flag'):
        ego_id = self.get_egoid_in_neighborhood_dataframe(outgoing_flag_column=outgoing_flag_column,
                                                          incoming_flag_column=incoming_flag_column)
        ego = (_T.ARTICLE, ego_id)
        nodes = set( [ (_T.ARTICLE, int(i)) for i in self.df[id_column].values ] )
        peers = set(nodes)
        peers.remove(ego)
        if not include_ego:
            nodes = peers
        if include_ego:
            links = set( [ (ego, peer) for peer in peers ] )
        else:
            links = set()
        for link in self.page_db.get_links_among(list(peers)): 
            links.add(link)
        return nodes, links


    def make_category_network_data_from_neighborhood_dataframe(self, id_column='id',
                                                               add_parents=True, category_filter=True):

        page_id_list = self.get_id_list()
        categories = set()
        nodes = set()
        for page_id in page_id_list:            
            cats = self.page_db.get_categories_of(page_id, name_filter=category_filter)
            if cats is None:
                print(f'category of {page_id} is None')
                continue
            # cats is (category_ids, category_names)
            for cat_id in cats[0]:
                categories.add( (_T.CATEGORY, cat_id) )
            for cat_name in cats[1]:
                nodes.add( (_T.CATEGORY, cat_name) )
        # 
        links_in_id = self.page_db.get_links_among(list(categories))
        return categories, nodes, self.page_db.stringify_link_data(links_in_id)
    
    
    def add_page_degree_columns(self, outgoing_column='outgoing_deg', incoming_column='incoming_deg'):
        self.df[outgoing_column] = [
            len(self.page_db.lookup_links('page', _T.ARTICLE, _T.ARTICLE, _T.FORWARD, i, []))
              for i in self.df['id'] ]
        self.df[incoming_column] = [
            len(self.page_db.lookup_links('page', _T.ARTICLE, _T.ARTICLE, _T.BACKWARD, i, []))
              for i in self.df['id'] ]
        return self

    
    def add_link_similarity_columns(self, embedding_type, 
                                 outgoing_column_base='outgoing_sim', incoming_column_base='incoming_sim'):
        if embedding_type == 'section':            
            embedding_function = self.get_section_page_vector
        elif embedding_type == 'wikipedia2vec':
            embedding_function = self.get_wikipedia2vec_page_vector
        else:
            raise Exception(f'Unknown embedding type "{embedding_type}"')
        #
        outgoing_column = f'{outgoing_column_base}_{embedding_type}'
        incoming_column = f'{incoming_column_base}_{embedding_type}'
        #
        self.df[outgoing_column] = [ self.get_average_link_similarity(i, embedding_function, direction=_T.FORWARD)
                                     for i in self.df['id'] ]
        self.df[incoming_column] = [ self.get_average_link_similarity(i, embedding_function, direction=_T.BACKWARD)
                                     for i in self.df['id'] ]
        return self

    
    def get_average_link_similarity(self, pid, embedding_function, direction=_T.FORWARD, centroid=False):
        #
        links = self.page_db.lookup_links('page', _T.ARTICLE, _T.ARTICLE, direction, pid, None)
        if links is None:
            return None
        v_list = []
        for i in links:
            v = embedding_function(i)
            if v is not None:
                v_list.append(v)
        if centroid:
            v0 = np.average(v_list, axis=0)
        else:
            v0 = embedding_function(pid)
            if v0 is None:
                return None
        return get_average_similarity(v0, v_list)
    
    
    def get_average_of(self, column):
        return np.average(self.df[column].values)

    
    def compare_columns(self, column1, column2):
        return np.count_nonzero((self.df[column1].values - self.df[column2].values) > 0) / len(self.df)

    
    def add_page_link_similarity_data(self, outgoing_column='outgoing_sim', incoming_column='incoming_sim'):
        pass
    
    
    # example: retain_data('redirect', 0) retains non-redirect pages
    #
    def retain_data(self, column, value):
        self.df = self.df[(self.df[column] == value)]
    
    
    def remove_undefined_data(self, column):
        self.df = self.df[~self.df[column].isnull()]

        
    def save_dataframe(self, filename):
        self.df.to_csv(filename)
    

    def read_dataframe(self, filename):
        self.df.read_csv(filename)
        
    
    def get_section_page_vector(self, q):
        return self.section_embedding.get_page_vector(q)

    
    def get_section_matrix(self, q):
        return self.section_embedding.get_section_matrix(q)
    
    
    def get_section_matrix_by_id(self, page_id):
        return self.section_embedding.get_section_matrix_by_id(page_id)
    
    
    def get_wikipedia2vec_page_vector(self, q):
        return self.wikipedia2vec.get_page_embedding(q)


    def make_filtered_dataframe(self, column, threshold):
        df1 = self.df[self.df[column] > threshold]
        return self.copy(df1)


    def make_neighborhood_network_data(self, include_ego=False):
        nodes, links = self.make_network_data_from_neighborhood_dataframe(include_ego=include_ego)
        return self.page_db.stringify_network_data_with_id_dict(nodes, links)


    def make_neighborhood_category_network_data(self, include_ego=False, add_parents=False, 
                                                expand_relations=False, category_filter=True):
        nodes, links = self.make_network_data_from_neighborhood_dataframe(include_ego=include_ego)
        self.page_db.add_categories_of_pages_to_network_data(nodes, links, name_filter=category_filter)
        if add_parents:
            self.page_db.add_parent_categories_to_network_data(nodes, links, expand_relations=expand_relations)
        return self.page_db.stringify_network_data_with_id_dict(nodes, links)

