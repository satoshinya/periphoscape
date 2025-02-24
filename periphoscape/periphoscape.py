import os
import sys
import json
from io import StringIO
import pandas as pd
from pathlib import Path
import networkx as nx
import urllib.parse
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.stats import chi2_contingency
from IPython.display import display, HTML

from periphoscape.constants import _T
from periphoscape.util import cos_similarity, filter_cluster_elements
from periphoscape.wikipedia_network import is_cat_name
from periphoscape.wikipedia_dataframe import *


class Periphoscape():
    default_params = {
        'threshold' : 0.9,
        'include_ego' : True,
        'category_filter' : True,
        'use_incoming' : True,
        'section_base' : True,
        'community_detection_method' : 'modularity',
        'centrality' : 'closeness',
        'weight' : 'wight',
        'early_break' : False,
        'least_number_of_articles' : 1,
        'significance_level' : 0.01,
        'chi2_lambda' : 0,
        'use_initial_adjacent_pages' : True,
        'naive_likelihood' : False,
        'clustering_metric' : 'euclidean',
        'clustering_method' : 'complete',
        'save_dir_top' : None,
        'save_data' : False,
        'IS_only' : False,
        'use_get_correlated_sections_x' : True,
        'full_html' : False
    }

    def __init__(self, page_name, page_db, wiki2vec, section_embedding):
        self.page_name = page_name
        self.page_category_ids = None
        self.page_category_names = None
        self.page_db = page_db
        self.wiki2vec = wiki2vec
        self.section_embedding = section_embedding
        self.wiki_df = None
        self.wiki_net = None
        self.communities = None
        #
        self.page_id = self.page_db.get_id_by_name(self.page_name)
        if self.page_id is None:
            raise Exception(f'Unable to get ID for "{self.page_name}"')
        if page_db.id2redirect[0].get(self.page_id) == 1:
            raise Exception(f'"{self.page_name}" is a redirect page')
        l = [x[1] for x in self.page_db.get_adjacent_elements(self.page_id, _T.ARTICLE,
                                                              linkdb=['page'],
                                                              adjacent_element_types=[_T.ARTICLE])]
        l.append(self.page_id)
        self.adjacent_pages = set(l)
        self.aspect_df = None
        self.aspect_html = None
        self.aspect_json = None
        self.params = Periphoscape.default_params
        self.supporting_evidence = {}

        
    def set_params(self, params):
        self.params.update(params)


    def initialize_dataframe(self):
        self.page_category_ids, self.page_category_names = self.page_db.get_categories_of(self.page_id)
        if self.page_id is not None:
            self.wiki_df = WikipediaDataFrame(self.page_db, self.wiki2vec, self.section_embedding)
            self.wiki_df.add_neighborhood_columns(self.page_id)
            self.wiki_df.retain_data('redirect', 0)
            self.wiki_df.add_ego_similarity_columns()
        return self


    # for backward compatibility
    def make_dataframe(self):
        return self.initialize_dataframe()

    
    def get_adjacent_pages_from_dataframe(self):
        return set(self.wiki_df.df['id'].values.tolist())

    
    def filter_dataframe(self, column='ego_sim_section', threshold=0.9):
        if self.wiki_df is not None:
            self.wiki_df = self.wiki_df.make_filtered_dataframe(column, threshold)
        return self

    
    def retain_only_outgoing_links(self):
        if self.wiki_df is not None:
            df0 = self.wiki_df.df
            df1 = df0[(df0['out_flag'] > 0) | ((df0['in_flag'] == 0) & (df0['out_flag'] == 0))]
            self.wiki_df = self.wiki_df.copy(df1)
        return self
        
        
    def make_neighborhood_article_category_network(self, include_ego=False, add_parents=False,
                                                   expand_relations=False, category_filter=True):
        if self.wiki_df is not None:
            self.wiki_net = WikipediaNetwork(self.wiki_df.make_neighborhood_category_network_data(
                                                            include_ego=include_ego, add_parents=add_parents,
                                                            expand_relations=expand_relations,
                                                            category_filter=category_filter))
        return self

    
    def add_edge_weights_to_network(self, default_weight=1.0, default_distance=1.0):
        if self.wiki_net is not None:
            links = list(self.wiki_net.g.edges)
            for link in links:
                link_in_id = (self.wiki_net.g.nodes[link[0]]['id'],
                              self.wiki_net.g.nodes[link[1]]['id'])
                weight = self.section_embedding.get_similarity(link_in_id)
                if weight is None:
                    weight = default_weight
                distance = self.section_embedding.get_distance(link_in_id)
                if distance is None:
                    distance = default_distance
                self.wiki_net.g.edges[link]['weight'] = weight
                self.wiki_net.g.edges[link]['distance'] = distance
    
    
    def get_network_communities(self, method, weight=None):
        if self.wiki_net is not None:
            self.communities = self.wiki_net.get_communities(method, weight=weight)
        return self

    
    def get_modularity_communities(self):
        if self.wiki_net is not None:
            self.communities = self.wiki_net.get_modularity_communities()
        return self
    
    
    def print_connected_component_summary(self):
        l = list(nx.connected_components(self.wiki_net.g))
        n = len(l)
        print(f'The network has {n} connected components:')
        for i in range(n):
            print(f'  [{i}] {len(l[i])} nodes')


    def retain_categories_in_communities(self):
        if self.communities is not None:
            self.communities = filter_cluster_elements(self.communities, is_cat_name)
        return self

    
    def exclude_page_categories_in_communities(self):
        if self.communities is not None:
            self.communities = [ 
                [x for x in y if x[2:] not in self.page_category_names] for y in self.communities
            ]
        return self    
    
    
    def get_centrality_of_communities(self, centrality, weight=None, top_n=None):
        if self.communities is None:
            return None
        return [ self.wiki_net.get_centrality_of_cluster_members(community, centrality, weight=weight)[:top_n]
                for community in self.communities ]

    
    def get_centrality_of_community_subnetworks(self, centrality, weight=None, top_n=None):
        if self.communities is None:
            return None
        return [ self.wiki_net.get_centrality_among_cluster_members(community, centrality, weight=weight, filter_=is_cat_name)[:top_n]
                    for community in self.communities ]

    
    def select_categories_of_communities(self):
        centrality = self.params['centrality']
        weight = self.params['weight']
        return [ self.select_categories_from_list(scored_list)
                 for scored_list in self.get_centrality_of_communities(centrality, weight=weight) ]
    
    
    def select_categories_of_community_subnetworks(self):
        centrality = self.params['centrality']
        weight = self.params['weight']
        return [ self.select_categories_from_list(scored_list)
                 for scored_list in self.get_centrality_of_community_subnetworks(centrality, weight=weight) ]

    
    def select_categories_from_list(self, scored_list):
        early_break = self.params['early_break']
        least_number_of_articles = self.params['least_number_of_articles']
        significance_level = self.params['significance_level']
        use_initial_adjacent_pages = self.params['use_initial_adjacent_pages']
        #
        selected_categories = set()
        found_articles = set()
        pages_for_test = self.adjacent_pages if use_initial_adjacent_pages else self.get_adjacent_pages_from_dataframe()
        for (node, score) in scored_list:
            node_id = self.wiki_net.g.nodes[node]['id'][1] # g.nodes[node]['id'] is (type, id) 
            articles = set(self.get_linked_articles(node))
            if len(articles - found_articles) == 0:
                if early_break:
                    break
                else:
                    continue
            if len(articles) >= least_number_of_articles:
                if ((significance_level is None) or
                    self.category_log_likelihood_test(node_id, pages_for_test)):
                    selected_categories.add(node)
                    found_articles.update(articles)
        return selected_categories

    
    def get_set_of_selected_category_names_by_likelihood(self):
        use_initial_adjacent_pages = self.params['use_initial_adjacent_pages']
        #
        pages = self.get_adjacent_pages_from_dataframe()
        pages_for_test = self.adjacent_pages if use_initial_adjacent_pages else self.get_adjacent_pages_from_dataframe()
        categories = set()
        for page in pages:
            cat_ids, cat_names = self.page_db.get_categories_of(page)
            categories.update(cat_ids)
        selected_category_names = set()
        for cat_id in categories:
            if self.category_log_likelihood_test(cat_id, pages_for_test):
                selected_category_names.add(self.page_db.get_category_name_by_id(cat_id))
        return selected_category_names
            
            
    def category_page_log_likelihood_test(self, cat_id, page_id):
        l = [x[1] for x in self.page_db.get_adjacent_elements(page_id, _T.ARTICLE,
                                                              linkdb=['page'],
                                                              adjacent_element_types=[_T.ARTICLE])]
        l.append(page_id)
        return self.category_log_likelihood_test(cat_id, set(l))


    def category_log_likelihood_test(self, cat_id, page_set):
        significance_level = self.params['significance_level']
        chi2_lambda = self.params['chi2_lambda']
        #
        cat_pages = set()
        for db in ['category', 'page']:
            l = self.page_db.links[db][_T.CATEGORY][_T.ARTICLE][_T.BACKWARD].get(cat_id, [])
            cat_pages.update(l)
        num_of_pages = self.page_db.get_num_of_pages()
        tbl = [
            [len(cat_pages & page_set), len(page_set - cat_pages)],
            [len(cat_pages - page_set), num_of_pages - len(cat_pages|page_set)]
        ]
        p = chi2_contingency(tbl, lambda_=chi2_lambda)[1]
        return (p < significance_level)

    
    def get_top_categories_of_communities(self, in_name=True):
        cats_list = []
        for community in self.communities:
            cat_ids = [ self.wiki_net.g.nodes[name]['id'][1] for name in community ]
            top_cats = self.page_db.get_top_categories_among(cat_ids)
            if in_name:
                top_cats = self.page_db.to_name(top_cats, type_=_T.CATEGORY)
            cats_list.append(top_cats)
        return cats_list
    
    
    def get_category_articles_in_json(self, cat):
        articles = []
        category_article_set = self.page_db.get_article_set_of_category(cat) ## category articles
        for page_id in self.wiki_df.df['id'].values:                        ## neighboring articles
            if page_id in category_article_set:
                title = self.page_db.get_name_by_id(page_id)
                jsn = { 'name' : title,
                         'type' : 'article',
                         'url'  : f'https://ja.wikipedia.org/wiki/{urllib.parse.quote(title)}'
                       }
                articles.append(jsn)
        return articles

    
    def convert_hierarchy_to_json(self, cat_list, hierarchy, parent=None):
        json_list = []
        for cat in cat_list:
            jsn = { 'name' : self.page_db.get_category_name_by_id(cat),
                     'type' : 'category' }
            json_list.append(jsn)
            subordinates = self.convert_hierarchy_to_json(hierarchy.get(cat, []), hierarchy, parent=cat)
            if len(subordinates) > 0:
                jsn['subordinates'] = subordinates
        if parent is not None:
            json_list.extend(self.get_category_articles_in_json(parent))
        return json_list

    
    def get_peripheral_hierarchy_in_json(self, save_as=None):
        json_list = []
        for community in self.communities:
            cat_ids = [ self.wiki_net.g.nodes[name]['id'][1] for name in community if
                        self.wiki_net.g.nodes[name]['type'] == 'category']
            top_list, hierarchy = self.page_db.get_hierarchy_among(cat_ids)
            jsn = self.convert_hierarchy_to_json(top_list, hierarchy)
            json_list.append(jsn)
        if save_as is not None:
            with open(save_as, 'w') as f:
                json.dump(json_list, f, ensure_ascii=False)
        return json_list
    

    def get_linked_articles_from_cluster(self, node_list):
        articles = set()
        for node in node_list:
            node_type = self.wiki_net.g.nodes[node]['id'][0]
            if node_type == _T.ARTICLE:
                continue
            for adjacent_article in self.get_linked_articles(node):
                articles.add(adjacent_article)
        return articles
    
    # ==> wikipedia_network
    def get_linked_articles(self, category_node):
        return [ adjacent_node for adjacent_node in self.wiki_net.g[category_node] if 
            self.wiki_net.g.nodes[adjacent_node]['id'][0] == _T.ARTICLE ]


    def get_community_articles(self):
        page_set_list = []
        for community in self.communities:
            page_set_list.append(self.get_linked_articles_from_cluster(community))
        return page_set_list
    
    
    def print_community_article_dependency(self):
        l = self.get_community_articles()
        n = len(l)
        for i in range(n):
            for j in range(i + 1, n):
                n_and = len(l[i] & l[j])
                r_i = n_and/len(l[i])
                r_j = n_and/len(l[j])
                print(f'{i}-{j}: {i}={r_i}  {j}={r_j}  {l[i] & l[j]}')


    ##################################################################################################

    def select_categories_based_on_global_centrality(self):
        threshold = self.params['threshold']
        include_ego = self.params['include_ego']
        category_filter = self.params['category_filter']        
        community_detection_method = self.params['community_detection_method']
        centrality = self.params['centrality']
        weight = self.params['weight']
        early_break = self.params['early_break']
        #
        self.filter_dataframe(column='ego_sim_section', threshold=threshold)
        self.make_neighborhood_article_category_network(include_ego=include_ego,
                                                        category_filter=category_filter)
        self.add_edge_weights_to_network()
        self.get_network_communities(community_detection_method, weight=None)
        self.retain_categories_in_communities()
        self.exclude_page_categories_in_communities()
        selected = self.select_categories_of_communities(centrality,
                                                         weight=weight,
                                                         early_break=early_break)
        #
        category_set = set()
        for categories in selected:
            for category in categories:
                category_set.add(self.wiki_net.g.nodes[category]['label'])
        return category_set


    def select_categories_based_on_subnet_centrality(self):
        threshold = self.params['threshold']
        criterion_column = self.params['criterion_column']
        include_ego = self.params['include_ego']
        category_filter = self.params['category_filter']
        use_incoming = self.params['use_incoming']
        community_detection_method = self.params['community_detection_method']
        weight = self.params['weight']
        naive_likelihood = self.params['naive_likelihood']
        #
        if not use_incoming:
            self.retain_only_outgoing_links() # retains also the EGO entry 
        self.filter_dataframe(column=criterion_column, threshold=threshold)
        if naive_likelihood:
            return self.get_set_of_selected_category_names_by_likelihood()
        self.make_neighborhood_article_category_network(include_ego=include_ego, category_filter=True)
        self.add_edge_weights_to_network()
        self.get_network_communities(community_detection_method, weight=weight)
        self.exclude_page_categories_in_communities()
        selected = self.select_categories_of_community_subnetworks()
        category_set = set()
        for categories in selected:
            for category in categories:
                category_set.add(self.wiki_net.g.nodes[category]['label'])
        return category_set

    
    def find_aspects(self):
        threshold = self.params['threshold']
        category_filter = self.params['category_filter']
        section_base = self.params['section_base']
        #
        self.initialize_dataframe()
        cat_df = pd.DataFrame()
        cat_ids = set()
        incoming = {}
        incoming_supporting_evidence = {}
        outgoing = {}
        section = {}
        section_supporting_evidence = {}
        pages_all = {}
        pages_sat = {}
        criterion_column = 'ego_sim_section' if section_base else 'ego_sim_page'
        self.set_params( { 'criterion_column' : criterion_column } )
        for page_id in self.wiki_df.df['id']:
            row = self.wiki_df.df[self.wiki_df.df['id'] == page_id]
            in_ = row['in_flag'].values[0]
            out_ = row['out_flag'].values[0]
            sec = 1 if (row['ego_sim_page'].values[0] <= threshold and
                        row['ego_sim_section'].values[0] > threshold) else 0
            sat = 1 if (threshold and row[criterion_column].values[0] > threshold) else 0
            ids, names = self.page_db.get_categories_of(page_id, name_filter=category_filter)
            cat_ids.update(ids)
            for id_ in ids:
                incoming[id_] = incoming.get(id_, 0) + in_
                if sat == 1 and in_ > 0:
                    if id_ not in incoming_supporting_evidence:
                        incoming_supporting_evidence[id_] = []
                    incoming_supporting_evidence[id_].append(page_id)
                outgoing[id_] = outgoing.get(id_, 0) + out_
                section[id_] = section.get(id_, 0) + sec
                if sat == 1 and sec > 0:
                    if id_ not in section_supporting_evidence:
                        section_supporting_evidence[id_] = []
                    section_supporting_evidence[id_].append(page_id)
                pages_all[id_] = pages_all.get(id_, 0) + 1
                pages_sat[id_] = pages_sat.get(id_, 0) + sat
        cat_df['id'] = list(cat_ids)
        cat_df['name'] = [ self.page_db.get_category_name_by_id(i) for i in cat_df['id'] ]
        cat_df['incoming'] = [ incoming[i] for i in cat_df['id'] ]
        cat_df['outgoing'] = [ outgoing[i] for i in cat_df['id'] ]
        cat_df['section']  = [ section[i] for i in cat_df['id'] ]
        cat_df['page_all'] = [ pages_all[i] for i in cat_df['id'] ]
        cat_df['page_sat'] = [ pages_sat[i] for i in cat_df['id'] ]
        #
        category_set = self.select_categories_based_on_subnet_centrality()
        #
        l = [ 1 if i in category_set else 0 for i in cat_df['name'] ]
        cat_df['selected'] = l
        #
        self.supporting_evidence['incoming'] = incoming_supporting_evidence
        self.supporting_evidence['section'] = section_supporting_evidence
        self.aspect_df = cat_df


    def get_aspect_dataframe(self):
        df = self.aspect_df
        return df[df['selected'] == 1]
        

    def generate_aspect_html(self):
        clustering_metric = self.params['clustering_metric']
        clustering_method = self.params['clustering_method']
        IS_only = self.params['IS_only']        
        #
        cat_df = self.aspect_df
        selected_category_ids = None
        if IS_only:
            selected_category_ids = cat_df[(cat_df['selected'] == 1) &
                                           (((cat_df['incoming'] > 0) & (cat_df['outgoing'] == 0)) |
                                            (cat_df['section'] == cat_df['page_sat']))]['id'].values
        else:
            selected_category_ids = cat_df[cat_df['selected'] == 1]['id'].values
        ##
        selected_category_id_update = []
        selected_category_vectors = []
        for selected_category_id in selected_category_ids:
            v = self.section_embedding.get_category_vector_x(selected_category_id)
            if v is None:
                print(f'** Cannot get vector for "{self.page_db.get_name_by_id(selected_category_id, type_=14)}"')
            else:
                selected_category_id_update.append(selected_category_id)
                selected_category_vectors.append(v.tolist())
        selected_category_id = selected_category_id_update
        ##
        rearranged_category_ids = selected_category_ids
        rearranged_category_vectors = selected_category_vectors
        ##
        n = len(selected_category_ids)
        if n > 2:
            Z = linkage(selected_category_vectors, metric=clustering_metric, method=clustering_method)
            leaves = leaves_list(Z)
            rearranged_category_ids = [ selected_category_ids[i] for i in leaves ]
            rearranged_category_vectors = [ selected_category_vectors[i] for i in leaves ]
        #
        html_data = json_data = None
        #
        if n > 0:
            html_data, json_data = self.to_html(rearranged_category_ids,
                                                rearranged_category_vectors,
                                                self.supporting_evidence.get('incoming'),
                                                self.supporting_evidence.get('section'))
            self.aspect_html = html_data
            self.aspect_json = json_data
    

    def to_html(self, selected_categories, category_vectors,
                incoming_supporting_evidence, section_supporting_evidence,
                hanging_indent_size='10px', anchor_spacing_size='10px'):
        use_get_correlated_sections_x = self.params['use_get_correlated_sections_x']
        full_html = self.params['full_html']
        #
        html_data = ''
        json_data = {}
        json_data['title'] = self.page_name
        json_data['categories'] = []
        hanging_indent = f'<span style="margin-right:{hanging_indent_size};"></span>'
        anchor_spacing = f'<span style="margin-right:{anchor_spacing_size};"></span>'
        with StringIO() as buf:
            if full_html:
                print('<html>\n<head>\n</head>', file=buf)
                print('<body>\n<dl>', file=buf)
            print('<div>', file=buf)                
            for category_list_index in range(len(selected_categories)):
                c_id = selected_categories[category_list_index]
                c_vector = category_vectors[category_list_index]
                c_data = {}
                json_data['categories'].append(c_data)
                c_data['name'] = self.page_db.get_category_name_by_id(c_id)
                c_data['anchor'] = self.make_category_anchor_of(c_data['name'])
                section_data_set = set()
                print(f'<dt>{c_data["anchor"]}</dt>', file=buf)
                for supporting_evidence in [incoming_supporting_evidence, section_supporting_evidence]:
                    for p_id in supporting_evidence.get(c_id, []):
                        if use_get_correlated_sections_x:
                            sections = self.get_correlated_sections_x(self.page_id, p_id)
                        else:
                            sections = self.get_correlated_sections(self.page_id, p_id)
                        if not self.is_trivial(sections[0]):
                            w = cos_similarity(c_vector, sections[0]['vector'])
                            section_data_set.add( (sections[0]['data_type'],
                                                   sections[0]['title'], sections[0]['header'], w) )
                        w = cos_similarity(c_vector, sections[1]['vector'])
                        section_data_set.add( (sections[1]['data_type'],
                                               sections[1]['title'], sections[1]['header'], w) )
                # sort section_data_set
                section_data_list = list(section_data_set)
                section_data_list.sort(key=lambda x: x[3], reverse=True)
                #
                c_data['sections'] = [ self.make_section_anchor_of(x, y, z)
                                       for x, y, z, w in section_data_list ]
                print(f'<dd>{hanging_indent}{anchor_spacing.join(c_data["sections"])}</dd>', file=buf)
            print('</dl>\n</div>', file=buf)
            if full_html:
                print('</body>\n</html>', file=buf)
            html_data = buf.getvalue()
        return html_data, json_data
        

    def display_aspects_in_notebook(self):
        if self.aspect_html is None:
            self.generate_aspect_html()
            return display(HTML(self.aspect_html))


    def save_aspect_data(self, save_dir_top):
        save_dir = f'{save_dir_top}/{self.page_name}'
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        if self.wiki_df:
            page_file = f'{save_dir}/pages.csv'
            self.wiki_df.df.to_csv(page_file)
        if self.aspect_df:
            category_file = f'{save_dir}/category.csv'
            self.aspect_df.to_csv(category_file)
        #
        if self.aspect_html:
            html_file = f'{save_dir}/aspect.html'
            with open(html_file, 'w') as f:
                print(self.aspect_html, file=f, end='')
        if self.aspect_json:
            json_file = f'{save_dir}/aspect.json'
            with open(json_file, 'w') as f:
                json.dump(self.aspect_json, f, ensure_ascii=False)


    def get_correlated_sections_x(self, p1_id, p2_id):
        r1 = self.section_embedding.search_agent.get_search_results(str(p1_id), size=100,
                                                    source=['title', 'header', 'data_type', 'vector'], 
                                                    fields=['wiki_id'])
        r2 = self.section_embedding.search_agent.get_search_results(str(p2_id), size=100,
                                                    source=['title', 'header', 'data_type', 'vector'], 
                                                    fields=['wiki_id'])
        max_score = -1
        max_pair = None
        for i in range(len(r1)):
            if r1[i]['data_type'] == 'page' or r1[i]['data_type'] == '_Lead':
                continue
            for j in range(len(r2)):
                if r2[j]['data_type'] == 'page' or r2[j]['data_type'] == '_Lead':
                    continue
                score = cos_similarity(r1[i]['vector'], r2[j]['vector'])
                if score > max_score:
                    max_score = score
                    max_pair = (i, j)
        return ( r1[max_pair[0]], r2[max_pair[1]] )


    # separator: &sdot; / &#149;
    #
    @staticmethod
    def make_section_anchor_of(data_type, title, header,
                               class_name = 'sectionAnchor',
                               url_base = 'https://ja.wikipedia.org/wiki/', 
                               separator=' / ', target='_aspect_info'):
        q_title = urllib.parse.quote(title)
        if data_type == 'page':
            anchor_text = title
            url = f'{url_base}{q_title}'
        elif header == '_Lead':
            anchor_text = title
            url = f'{url_base}{q_title}#'
        else:
            anchor_text = f'{title}{separator}{header}'
            q_header = urllib.parse.quote(header)
            url = f'{url_base}{q_title}#{q_header}'
        #
        return f'<a href="{url}" class="{class_name}" target="{target}">{anchor_text}</a>'

    
    @staticmethod
    def make_category_anchor_of(title,
                                class_name='categoryAnchor',
                                url_base='https://ja.wikipedia.org/wiki/',
                                target='_aspect_info'):
        q_title = urllib.parse.quote(title)
        url = f'{url_base}Category:{q_title}'
        return f'<a href="{url}" class="{class_name}" target="{target}">{title}</a>'

    
    @staticmethod
    def is_trivial(section_data):
        return section_data['data_type'] == 'page' or section_data['header'] == '_Lead'

