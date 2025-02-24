import os
import sys
import re
import pprint
import networkx as nx
import matplotlib.pyplot as plt

from periphoscape.constants import _T


class WikipediaSearch():
    default_source = ["title", "wiki_id", "category", "outgoing_link"]
    default_fields = ["title^2", "text"]
    category_exclude_pattern = re.compile('[あ-ん]ページ|[あ-ん]記事|[あ-ん]項目|出典を必要とする|スタブ|曖昧さ回避|ウェイバックリンク')
    
    def __init__(self, client):
        self.client = client
        self.font_family = None

    @classmethod
    def convert_result(cls, data, source_name):
        if source_name == 'wiki_id':
            return int(data)
        elif source_name == 'category':
            return cls.filter_category(data)
        else:
            return data
        
    @classmethod
    def filter_category(cls, category_list):
        return [ x for x in category_list if not cls.category_exclude_pattern.search(x)]
            
    def get_search_results(self, query_text, size=100, source=None, fields=None):
        #
        all_sources = ['wiki_id', 'category', 'title', 'header', 'data_type', 'vector', 'outgoing_link',
                       'redirect', 'wikibase_item', 'text']
        #
        if not source:
            source = WikipediaSearch.default_source
        if not fields:
            fields = WikipediaSearch.default_fields
        q = self.client.make_multi_match_query([query_text], [], size=size,
                                               #source=source,
                                               source=all_sources,
                                               fields=fields, 
                                               query_type="phrase")
        results_of_elasticsearch = self.client.do_search(q)
        ##
        #Trace.add(query_text, results_of_elasticsearch)
        ##
        r = [ { x : WikipediaSearch.convert_result(r['_source'][x], x) for x in source }
                    for r in results_of_elasticsearch['hits']['hits'] ]
        return r
        #
        #return [ {'wiki_id'       : int(r['_source']['wiki_id']),
        #          'title'         : r['_source']['title'],
        #          'category'      : r['_source']['category'],
        #          'outgoing_link' : r['_source']['outgoing_link'] }
        #           for r in results['hits']['hits'] ]
        
    def make_result_network(self, results):
        #g = nx.DiGraph()
        g = nx.Graph()
        articles = [ r['title'] for r in results ]
        for a in articles:
            g.add_node(a, color='lightblue', type='article', label=a)
        for r in results:
            n0 = r['title']
            for c in r['category']:
                cname = f'C:{c}'
                if not cname in g:
                    g.add_node(cname, color='orange', type='category', label=c)
                g.add_edge(n0, cname, type='a2c')
            for n1 in r['outgoing_link']:
                if n1 in articles:
                    g.add_edge(n0, n1, type='a2a')
        return g
    

    def results_to_network_data(self, results):
        nodes = set()
        links = set()
        articles = [ r['title'] for r in results ]
        for r in results:
            n0 = (_T.ARTICLE, r['title'])
            nodes.add(n0)
            for c in r['category']:
                n1 = (_T.CATEGORY, c)
                nodes.add(n1)
                links.add( (n0, n1) )
            for p in r['outgoing_link']:
                if p not in articles:
                    continue
                n1 = (_T.ARTICLE, p)
                nodes.add(n1)
                links.add( (n0, n1) )
        return nodes, links
    
    
    def get_search_result_network(self, query_text, size=100, source=None, fields=None):
        r = self.get_search_results(query_text, size=size, source=source, fields=fields)
        return self.make_result_network(r)

    
    def get_search_result_network_data(self, query_text, size=100, source=None, fields=None):
        r = self.get_search_results(query_text, size=size, source=source, fields=fields)
        return self.results_to_network_data(r)


