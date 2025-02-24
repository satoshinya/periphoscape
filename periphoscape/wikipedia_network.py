import os
import sys
import platform
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities, label_propagation_communities, asyn_lpa_communities, louvain_communities
import matplotlib.pyplot as plt

from periphoscape.constants import _T


community_dection_functions = {
    'modularity' : greedy_modularity_communities,
    'label'      : asyn_lpa_communities,
    'louvain'    : louvain_communities
}


def get_japanese_font_family():
    osname = platform.platform()
    if osname.startswith('macOS'):
        return 'Hiragino Sans'
    if osname.startswith('Linux'):
        return 'Noto Sans CJK JP'
    return None


def count_edges_by_type(g, node):
    count = { _T.ARTICLE : 0, _T.CATEGORY : 0 }
    for adjacent_node in g[node]:
        type_ = g.nodes[adjacent_node]['id'][0]
        count[type_] += 1
    return count


def article_degree_centrality(g):
    return { n : count_edges_by_type(g, n)[_T.ARTICLE] for n in g.nodes }


def composite_degree_centrality(g):
    centrality = {}
    for n in g.nodes:
        count = count_edges_by_type(g, n)
        score = (count[_T.ARTICLE] + 1) * (count[_T.CATEGORY] + 1)
        centrality[n] = score
    return centrality


network_centrality_functions = {
    'degree'      : lambda x, weight=None: nx.degree_centrality(x),
    'eigenvector' : nx.eigenvector_centrality,
    'closeness'   : lambda x, weight=None: nx.closeness_centrality(x) if weight is None else nx.closeness_centrality(x, distance='distance'),
    'betweenness' : nx.betweenness_centrality,
    'subgraph'    : nx.subgraph_centrality,    
    'harmonic'    : nx.harmonic_centrality,
    'article'     : lambda x, weight=None: article_degree_centrality(x),
    'composite'   : lambda x, weight=None: composite_degree_centrality(x),
    '_default_'   : nx.degree_centrality
}


def is_cat_name(name):
    return name[:2] == 'c:'


class WikipediaNetwork():
    P = {
        _T.ARTICLE : {
            'name'   : 'article',
            'color'  : 'lightblue',
            'prefix' : 'a',
            'link_type' : {
                _T.ARTICLE  : 'a2a',
                _T.CATEGORY : 'a2c'
            }
         },
        _T.CATEGORY : {
            'name'   : 'category',
            'color'  : 'orange',
            'prefix' : 'c',
            'link_type' : {
                _T.ARTICLE  : 'c2a',
                _T.CATEGORY : 'c2c'
            }
        }
    }
    
    
    def __init__(self, g_or_data, font_family=None):
        if type(g_or_data) == nx.classes.graph.Graph:
            self.g = g_or_data
        else:
            self.g = WikipediaNetwork.create_network_from_data(g_or_data)
        self.font_family = font_family
        if self.font_family is None:
            self.font_family = get_japanese_font_family()

        
    @staticmethod
    def create_network_from_data(network_data):
        nodes = network_data[0]
        links = network_data[1]
        attribue_dict = {}
        if len(network_data) == 3:
            attribue_dict = network_data[2]
        g = nx.Graph()
        for n in nodes:
            type_id = n[0]
            type_name = WikipediaNetwork.P[type_id]['name']
            node_name = f"{WikipediaNetwork.P[type_id]['prefix']}:{n[1]}"                
            label = n[1]
            color = WikipediaNetwork.P[type_id]['color']
            attributes = attribue_dict.get(n, {})
            g.add_node(node_name, color=color, type=type_name, label=label, **attributes)
        for l in links:
            n0 = l[0]
            n0_type_id = n0[0]
            n0_node_name = f"{WikipediaNetwork.P[n0_type_id]['prefix']}:{n0[1]}"
            n1 = l[1]
            n1_type_id = n1[0]
            n1_node_name = f"{WikipediaNetwork.P[n1_type_id]['prefix']}:{n1[1]}"
            type_name = WikipediaNetwork.P[n0_type_id]['link_type'][n1_type_id]
            g.add_edge(n0_node_name, n1_node_name, type=type_name)
        return g
        
    
    def set_font_family(self, ff):
        self.font_family = ff
    
    
    def get_node_colors_from_attribues(self, attribute_name='color'):
        return [node[attribute_name] for node in self.g.nodes.values()]
    
    
    def get_node_labels_from_attribues(self, attribute_name='label', label_filter=None):
        if label_filter:
            return { k : label_filter(v[attribute_name]) for (k, v) in self.g.nodes.items() }
        return { k : v[attribute_name] for (k, v) in self.g.nodes.items() }
    
    
    def draw_network(self, node_colors=None, node_labels=None, font_family=None):
        if not node_colors:
            #node_colors = 'lightblue'
            node_colors = self.get_node_colors_from_attribues()
        if not node_labels:
            # node_labels = {i : str(i) for i in self.g.nodes.keys()}
            node_labels = self.get_node_labels_from_attribues()
        if not font_family:
            font_family = self.font_family
        fig = plt.figure(figsize=(15, 15))
        #pos = nx.kamada_kawai_layout(g)
        pos = nx.spring_layout(self.g, k=0.3)
        nx.draw_networkx_edges(self.g, pos, alpha=0.4, edge_color="black", width=1)
        nx.draw_networkx_nodes(self.g, pos, node_color=node_colors, alpha=0.6, node_size=200)
        if font_family:
            nx.draw_networkx_labels(self.g, pos, node_labels, font_size=12, 
                                    font_family=font_family)
        else:
            nx.draw_networkx_labels(self.g, pos, node_labels, font_size=12)
        plt.axis("off")
    
    
    def get_components(self, types=None):
        if not types:
            types = ['category', 'article']
        if self.g.is_directed():
            cc = nx.strongly_connected_components(self.g)
        else:
            cc = nx.connected_components(self.g)
        components = []
        for c in cc:
            r = [i for i in c if self.g.nodes[i]['type'] in types]
            if r:
                components.append(r)
        return components
    
    
    def get_max_component(self, types=None):
        return max(self.get_components(types=types), key=len)
    
    
    def get_max_component_as_network(self, types=None):
        return self.g.subgraph(self.get_max_component(types=types)).copy()
    

    # component is a subset of nodes in self.g
    # from which links are inherited.
    #
    def get_wikipedia_network_of_component(self, component):        
        return WikipediaNetwork(self.g.subgraph(component).copy(), font_family=self.font_family)
    
    
    def get_max_component_as_wikipedia_network(self, types=None):
        component = self.get_max_component(types=types)
        return self.get_wikipedia_network_of_component(component)
    
    
    def get_modularity_communities(self):
        return greedy_modularity_communities(self.g)

    
    def get_communities(self, method, weight=None):
        detection_function = community_dection_functions[method]
        return detection_function(self.g, weight=weight)


    def get_centrality_of_cluster_members(self, cluster, centrality_method, weight=None, top_n=None):
        centrality_function = network_centrality_functions[centrality_method]
        centrality = [n for n in centrality_function(self.g, weight=weight).items() if n[0] in cluster]
        if top_n is None:
            return sorted(centrality, key=lambda x: x[1], reverse=True)
        else:
            return sorted(centrality, key=lambda x: x[1], reverse=True)[:top_n]            
    
    
    def get_closeness_of_cluster_members(self, cluster, top_n=None):
        closeness = [n for n in nx.closeness_centrality(self.g).items() if n[0] in cluster]
        if top_n is None:
            return sorted(closeness, key=lambda x: x[1], reverse=True)
        else:
            return sorted(closeness, key=lambda x: x[1], reverse=True)[:top_n]

        
    def get_betweenness_of_cluster_members(self, cluster):
        betweenness = [n for n in nx.betweenness_centrality(self.g).items() if n[0] in cluster]
        return sorted(betweenness, key=lambda x: x[1], reverse=True)

    
    def get_centrality_among_cluster_members(self, cluster, centrality_method, weight=None, filter_=None):
        centrality_function = network_centrality_functions[centrality_method]
        if filter_ is None:
            centrality = [n for n in centrality_function(self.g.subgraph(cluster), weight=weight).items()]
        else:
            centrality = [n for n in centrality_function(self.g.subgraph(cluster), weight=weight).items() if filter_(n[0])]
        return sorted(centrality, key=lambda x: x[1], reverse=True)

    
    def get_closeness_among_cluster_members(self, cluster, filter_=None):
        if filter_ is None:
            closeness = [n for n in nx.closeness_centrality(self.g.subgraph(cluster)).items()]
        else:
            closeness = [n for n in nx.closeness_centrality(self.g.subgraph(cluster)).items() if filter_(n[0])]
        return sorted(closeness, key=lambda x: x[1], reverse=True)


    def get_betweenness_among_cluster_members(self, cluster, filter_=None):
        if filter_ is None:
            betweenness = [n for n in nx.betweenness_centrality(self.g.subgraph(cluster)).items()]
        else:
            betweenness = [n for n in nx.betweenness_centrality(self.g.subgraph(cluster)).items() if filter_(n[0])]
        return sorted(betweenness, key=lambda x: x[1], reverse=True)


    def filter_by_closeness(self, threashold):
        l = [ i[0] for i in nx.closeness_centrality(self.g).items() if i[1] > threashold ]
        return self.get_wikipedia_network_of_component(l)

    
    def filter_by_betweenness(self, threashold):
        l = [ i[0] for i in nx.betweenness_centrality(self.g).items() if i[1] > threashold ]
        return self.get_wikipedia_network_of_component(l)
    
