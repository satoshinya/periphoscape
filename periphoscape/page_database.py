import os
import re
import csv
import pickle

from periphoscape.constants import _T
#from periphoscape.dataset import Dataset

class WikipediaPageDB():
    category_exclude_clues = [
        'ページ',
        '記事',
        '項目',
        '座標',
        '識別子',
        '人物',
        'ウィキデータ',
        '出典を必要とする',
        'スタブ',
        '曖昧さ回避',
        'ウェイバックリンク',
        '参照方法',
        'E_number_from_Wikidata'
    ]
    category_exclude_pattern = re.compile('|'.join(category_exclude_clues))
    
    def __init__(self, dataset, page_types=None):
        self.dataset = dataset
        #
        self.default_target_page_types = [_T.ARTICLE, _T.CATEGORY]
        #
        self.target_page_types = page_types
        if not self.target_page_types:
            self.target_page_types = self.default_target_page_types
        #
        self.id2name = None
        self.name2id = None
        self.links = None
        self.max_id = {_T.ARTICLE : 0, _T.CATEGORY : 0}
        self.num_of_pages = None

        
    def id2name_is_ready(self):
        return self.id2name is not None

    
    def init_db(self):
        self.id2name = { type_ : {} for type_ in self.target_page_types }
        self.name2id = { type_ : {} for type_ in self.target_page_types }
        self.id2redirect = { type_ : {} for type_ in self.target_page_types }

        
    def read_page_csv(self):
        self.init_db()
        with open(self.dataset.files['page']) as f:
            reader = csv.reader(f)
            for row in reader:
                id_ = int(row[0])
                type_ = int(row[1])
                name = row[2]
                redirect = int(row[3])
                if type_ not in self.target_page_types:
                    continue
                self.id2name[type_][id_] = name
                self.name2id[type_][name] = id_
                self.id2redirect[type_][id_] = redirect
                if id_ > self.max_id[type_]:
                    self.max_id[type_] = id_
        #
        if _T.CATEGORY not in self.target_page_types:
            return
        with open(self.dataset.files['category']) as f:
            reader = csv.reader(f)
            for row in reader:
                id_ = int(row[0])
                name = row[1]
                if name not in self.name2id[_T.CATEGORY]:
                    self.id2name[_T.CATEGORY][id_] = name
                    self.name2id[_T.CATEGORY][name] = id_
                    if id_ > self.max_id[_T.CATEGORY]:
                        self.max_id[_T.CATEGORY] = id_

    
    def get_max_page_id(self):
        return self.max_id[_T.ARTICLE]

    
    def get_max_category_id(self):
        return self.max_id[_T.CATEGORY]

    
    def to_name(self, ids, type_=_T.ARTICLE):
        if isinstance(ids, int):
            return self.lookup_id2name(type_, ids)
        if isinstance(ids, list):
            return [self.lookup_id2name(type_, i) for i in ids]
        return None


    def stringify_network_data_with_id_dict(self, nodes_in_id, links_in_id):
        nodes, id_dict = self.stringify_node_data_with_id_dict(nodes_in_id)
        links = self.stringify_link_data(links_in_id)
        return nodes, links, id_dict

    
    def stringify_node_data_with_id_dict(self, nodes_in_id):
        nodes = set()
        id_dict = {}
        for node in nodes_in_id:
            node_type, node_id = node
            #s = self.id2name[node_type].get(node_id)
            s = self.id2name[node_type].get(node_id)            
            if s is None:
                s = str(node_id)
            str_node = (node_type, s)
            nodes.add(str_node)
            id_dict[str_node] = { 'id' : node }
        return nodes, id_dict

    
    def stringify_network_data(self, nodes_in_id, links_in_id):
        return self.stringify_node_data(nodes_in_id), self.stringify_link_data(links_in_id)

    
    def stringify_node_data(self, nodes_in_id):
        nodes = set()
        for node in nodes_in_id:
            node_type, node_id = node
            s = self.lookup_id2name(node_type, node_id)
            if s is None:
                s = str(node_id)
            nodes.add( (node_type, s) )
        return nodes

    
    def stringify_link_data(self, links_in_id):
        links = set()
        for link in links_in_id:
            (node1_type, node1_id), (node2_type, node2_id) = link
            s1 = self.lookup_id2name(node1_type, node1_id)
            if s1 is None:
                s1 = str(node1_id)
            s2 = self.lookup_id2name(node2_type, node2_id)
            if s2 is None:
                s2 = str(node2_id)
            links.add( ((node1_type, s1), (node2_type, s2)) )
        return links

    
    # self.links[source_type][destinaton_type][forward/backward][from_id] = [to_id_1, to_id_2, ....]
    def init_links(self):
        self.links = {}
        for link_type in ['category', 'page']:
            self.links[link_type] = {}
            for type1 in self.target_page_types:
                self.links[link_type][type1] = {}
                for type2 in self.target_page_types:
                    self.links[link_type][type1][type2] = { _T.FORWARD : {}, _T.BACKWARD : {} }


    def add_link(self, link_type, source_id, source_type, destination_id, destination_type):
        d = self.links[link_type][source_type][destination_type][_T.FORWARD]
        if source_id not in d:
            d[source_id] = []
        if destination_id not in d[source_id]:
            d[source_id].append(destination_id)
        #
        # reverse link
        d = self.links[link_type][destination_type][source_type][_T.BACKWARD]
        if destination_id not in d:
            d[destination_id] = []
        if source_id not in d[destination_id]:
            d[destination_id].append(source_id)


    def read_categorylinks(self):
        if self.dataset.files['categorylinks'].endswith('pickle'):
            self.load_categorylinks_pickle()
        else:
            self.read_categorylinks_csv()        
        

    def read_categorylinks_csv(self):
        if not self.id2name_is_ready():
            self.read_page_csv()
        #
        with open(self.dataset.files['categorylinks']) as f:
            reader = csv.reader(f)
            for row in reader:
                source_id = int(row[0])
                destination_name = row[1]
                destination_id = self.name2id[_T.CATEGORY].get(destination_name)
                source_type = _T.D.get(row[2])
                #
                if source_type is None:
                    continue
                if destination_id:
                    self.add_link('category', source_id, source_type, destination_id, _T.CATEGORY)
                else:
                    pass
                    #print(f'** category "{destination_name}" not found in page DB')


    def read_pagelinks(self):
        if self.dataset.files['pagelinks'].endswith('pickle'):
            self.load_pagelinks_pickle()
        else:
            self.read_pagelinks_csv()

    
    def read_pagelinks_csv(self):
        if not self.id2name_is_ready():
            self.read_page_csv()
        #
        with open(self.dataset.files['pagelinks']) as f:
            reader = csv.reader(f)
            for row in reader:
                source_id = int(row[0])
                destination_type = int(row[1])
                if destination_type not in self.target_page_types:
                    continue
                destination_name = row[2]
                destination_id = self.name2id[destination_type].get(destination_name)         
                source_type = int(row[3])
                if source_type not in self.target_page_types:
                    continue
                if destination_id:
                    self.add_link('page', source_id, source_type, destination_id, destination_type)
                else:
                    pass
                    #print(f'** "{destination_name}" (type {destination_type}) not found in page DB')


    def save_pagelinks_as_pickle(self, out_filename):
        if os.path.exists(out_filename):
            print(f"File {out_filename} already exists")
        else:
            with open(out_filename, mode='wb') as fout:
                pickle.dump(self.links['page'], fout)
    
    
    def save_categorylinks_as_pickle(self, out_filename):
        if os.path.exists(out_filename):
            print(f"File {out_filename} already exists")
        else:
            with open(out_filename, mode='wb') as fout:
                pickle.dump(self.links['category'], fout)
      

    def load_pagelinks_pickle(self):
        with open(self.dataset.files['pagelinks'], mode='br') as fin:
            self.links['page'] = pickle.load(fin)


    def load_categorylinks_pickle(self):
        with open(self.dataset.files['categorylinks'], mode='br') as fin:
            self.links['category'] = pickle.load(fin)
            
                
    def get_name_by_id(self, id_, type_=_T.ARTICLE, pass_through_string=False):
        if pass_through_string and isinstance(id_, str):
            return id_
        return self.lookup_id2name(type_, id_)

    
    def get_category_name_by_id(self, id_):
        return self.lookup_id2name(_T.CATEGORY, id_)

    
    def get_id_by_name(self, name, type_=_T.ARTICLE):
        return self.lookup_name2id(type_, name)

    
    def get_id_and_name(self, id_or_name, type_=_T.ARTICLE):
        id_ = id_or_name
        name = id_or_name
        if isinstance(id_or_name, int):
            name = self.lookup_id2name(type_, id_)
        else:
            id_ = self.lookup_name2id(type_, name)
        return id_, name

    
    ### LOOKUP
    
    def lookup_id2name(self, type_, id_, default=None):
        q = (type_, id_)
        r = self.id2name[type_].get(id_, default)
        #Trace.add( q, r )
        return r

    
    def lookup_name2id(self, type_, name, default=None):
        q = (type_, name)
        r = self.name2id[type_].get(name, default)
        #Trace.add( q, r )
        return r

    
    def lookup_id2redirect(self, type_, id_, default=None):
        q = (type_, id_)
        r = self.id2redirect[type_].get(id_, default)
        #Trace.add( q, r )
        return r

    
    def lookup_links(self, db, this_type, peer_type, direction, this_id, default):
        q = (db, this_type, peer_type, direction, this_id)
        r = self.links[db][this_type][peer_type][direction].get(this_id, default)
        #Trace.add( q, r )
        return r

    
    def get_num_of_pages(self):
        return  len(self.id2name[0]) if (self.num_of_pages is None) else self.num_of_pages

    
    def set_num_of_pages(self, n):
        self.num_of_pages = n
    
    ###
    
    
    # get_adjacent_elements(linkdb=['page'], this_id=123, this_type=_T.ARTICLE,
    #                       directions=[_T.FORWARD], adjacent_element_types=[_T.ARTICLE])
    #
    def get_adjacent_elements(self, this_id, this_type,
                              linkdb=None, directions=None, adjacent_element_types=None):
        if linkdb is None:
            linkdb = ['page', 'category']
        if directions is None:
            directions = [_T.FORWARD, _T.BACKWARD]
        if adjacent_element_types is None:
            adjacent_element_types = [_T.ARTICLE, _T.CATEGORY]
        elements = []
        this = (this_type, this_id)
        for db in linkdb:
            for adjacent_element_type in adjacent_element_types:
                for direction in directions:
                    elements.extend([ (adjacent_element_type, i) for i in
                        self.lookup_links(db, this_type, adjacent_element_type, direction, this_id, []) ])
        #
        return elements


    def get_article_set_of_category(self, cat_id):
        articles = set()
        for db in ['page', 'category']:
            for page_id in self.lookup_links(db, _T.CATEGORY, _T.ARTICLE, _T.BACKWARD, cat_id, []):
                articles.add(page_id)
        #
        return articles

    
    # network of nodes linked to/from (ego_type, ego_id)
    #
    def get_neighborhood_network_data(self, ego_id, ego_type,
                                      linkdb=None, directions=None, adjacent_element_types=None):
        ego = (ego_type, ego_id)
        neighbors = self.get_adjacent_elements(ego_id, ego_type, linkdb=linkdb, directions=directions,
                                               adjacent_element_types=adjacent_element_types)
        links = self.get_links_among(neighbors)
        for neighbor in neighbors:
            links.add( (ego, neighbor) )
        neighbors.append(ego)
        return set(neighbors), links

    
    # this = (this_type, this_id)
    #
    def has_wikilink_between(self, this, that, linkdb=None):
        if linkdb is None:
            linkdb = ['page', 'category']
        results = []
        this_type, this_id = this
        that_type, that_id = that
        #Trace.add( (this, that, linkdb), None )
        for db in linkdb:
            for direction in [_T.FORWARD, _T.BACKWARD]:
                if that_id in self.lookup_links(db, this_type, that_type, direction, this_id, []):
                    return True
        return False

    
    # get pages linked from/to the specified page.
    #
    def get_wikilink_ends(self, page_title_or_id, to_name=True):
       page_id, page_name = self.get_id_and_name(page_title_or_id)
       if page_id is None or page_name is None:
           return None
       outgoing = [ i[1] for i in
                    self.get_adjacent_elements(page_id, _T.ARTICLE,
                                               linkdb=['page'],
                                               directions=[_T.FORWARD],
                                               adjacent_element_type=_T.ARTICLE) ]
       incoming = [ i[1] for i in
                    self.get_adjacent_elements(page_id, _T.ARTICLE,
                                               linkdb=['page'],
                                               directions=[_T.BACKWARD],
                                               adjacent_element_type=_T.ARTICLE) ]
       if to_name:
           outgoing = [ self.get_name_by_id(i) for i in outgoing
                        if self.get_name_by_id(i) is not None ]
           incoming = [ self.get_name_by_id(i) for i in incoming
                        if self.get_name_by_id(i) is not None ]
       return outgoing, incoming

   
    # element_list should be like [ (type1, id1), (type2, id2), ... ]
    #
    def get_links_among(self, element_list):
        n = len(element_list)
        links = set()
        for i in range(n):
            ei = element_list[i]
            for j in range(i, n):
                ej = element_list[j]
                if self.has_wikilink_between(ei, ej):
                    links.add( (ei, ej) )
        return links

    
    def get_network_data_from(self, element_list):
        return set(element_list), self.get_links_among(element_list)

    
    def get_categories_of(self, page_title_or_id, name_filter=True):
        page_id, page_name = self.get_id_and_name(page_title_or_id)
        if page_id is None or page_name is None:
            return None
        cat_names = set()
        cat_ids = set()        
        for cid in self.lookup_links('category', _T.ARTICLE, _T.CATEGORY, _T.FORWARD, page_id, []):
            cname = self.get_name_by_id(cid, type_=_T.CATEGORY)
            if cname is not None:
                if (not name_filter) or (not WikipediaPageDB.category_exclude_pattern.search(cname)):
                    cat_names.add(cname)
                    cat_ids.add(cid)
        for cid in self.lookup_links('page', _T.ARTICLE, _T.CATEGORY, _T.FORWARD, page_id, []):
            cname = self.get_name_by_id(cid, type_=_T.CATEGORY)
            if cname is not None:
                if (not name_filter) or (not WikipediaPageDB.category_exclude_pattern.search(cname)):
                    cat_names.add(cname)
                    cat_ids.add(cid)
        return cat_ids, cat_names

    
    # node = (node_type, node_id), node_type should be _T.ARTICLE.
    # link = (node1, node2) = (  (node1_type, node1_id),  (node2_type, node2_id) )
    #
    def add_categories_of_pages_to_network_data(self, nodes, links, name_filter=True):
        # add categories,  and links from each page to its categories
        categories_to_add = set()
        for page in nodes:
            page_id = page[1]
            category_ids, category_names = self.get_categories_of(page_id, name_filter=name_filter)
            for category_id in category_ids:
                category =  (_T.CATEGORY, category_id)
                categories_to_add.add(category)
                links.add( (page, category) )
        for category in categories_to_add:
            nodes.add(category)
        # add links among added categories
        for link in self.get_links_among(list(categories_to_add)):
            links.add(link)

            
    def add_parent_categories_to_network_data(self, nodes, links, name_filter=True, expand_relations=False):
        category_ids = [ i[1] for i in nodes if i[0] == _T.CATEGORY ]
        added_categories = set()
        for cat_id in category_ids:
            parents = self.get_parent_categories(cat_id)
            if parents is None:
                continue
            for p_id in parents:
                p_name = self.get_name_by_id(p_id, type_=_T.CATEGORY)
                if p_name is None:
                    continue
                if name_filter and WikipediaPageDB.category_exclude_pattern.search(p_name):
                    continue
                parent = (_T.CATEGORY, p_id)
                nodes.add(parent)
                links.add( ((_T.CATEGORY, cat_id), parent) )
                added_categories.add(parent)
        #
        if expand_relations:
            for link in self.get_links_among(added_categories):
                links.add(link)

    
    def get_parent_categories(self, cat_id):
        return self.lookup_links('category', _T.CATEGORY, _T.CATEGORY, _T.FORWARD, cat_id, None)

    
    def get_child_categories(self, cat_id):
        return self.lookup_links('category', _T.CATEGORY, _T.CATEGORY, _T.BACKWARD, cat_id, None)


    # Starting from cat_list, trace the entire category network upward
    # and extract the dead-end elements (i.e., those without a parent).
    def get_top_categories_of(self, cat_list):
        to_check = cat_list[:]
        checked = set()
        roots = set()
        while len(to_check) > 0:
            c = to_check.pop()
            checked.add(c)
            parents = self.get_parent_categories(c)
            if parents is None or len(parents) == 0:
                roots.add(c)
                continue
            for p in parents:
                if p in checked or p in to_check or p in roots:
                    continue
                to_check.append(p)
        return roots


    # Extract the elements that have no upper element in cat_list.
    def get_top_categories_among_0(self, cat_list):
        candidates = cat_list[:]
        checked = set()
        for candidate in cat_list:
            check_for_candidate_is_done = False
            to_check = [candidate]
            to_check_next = set()
            while len(to_check) > 0:
                for c in to_check:
                    checked.add(c)
                    parents = self.get_parent_categories(c)
                    if parents is None or len(parents) == 0:
                        #print(f'{self.get_name_by_id(c, type_=_T.CATEGORY)}({c}) has no parents')
                        continue
                    for parent in parents:
                        if parent in cat_list:
                            #print(f'{self.get_name_by_id(c, type_=_T.CATEGORY)}({c}) has parent ' 
                            #      f'{self.get_name_by_id(parent, type_=_T.CATEGORY)}')
                            candidates.remove(candidate)
                            check_for_candidate_is_done = True
                            break
                        else:
                            if parent not in checked:
                                to_check_next.add(parent)
                    if check_for_candidate_is_done:
                        break
                if check_for_candidate_is_done:
                    break
                to_check = list(to_check_next)
                to_check_next.clear()
        return candidates

    
    def get_closest_ancestor_among(self, cat, cat_list):
        checked = set()
        to_check = [cat]
        to_check_next = set()
        while len(to_check) > 0:
            for c in to_check:
                checked.add(c)
                parents = self.get_parent_categories(c)
                if parents is None or len(parents) == 0:
                    continue
                for parent in parents:
                    if parent in cat_list:
                        return parent
                    else:
                        if parent not in checked:
                            to_check_next.add(parent)
            to_check = list(to_check_next)
            to_check_next.clear()
        return cat

    
    def get_top_categories_among(self, cat_list):    
        top_categories = []
        for cat in cat_list:
            ancestor = self.get_closest_ancestor_among(cat, cat_list)
            if ancestor == cat:
                top_categories.append(cat)
        return top_categories

    
    def get_hierarchy_among(self, cat_list):
        top_categories = []
        hierarchy = {}
        for cat in cat_list:
            ancestor = self.get_closest_ancestor_among(cat, cat_list)            
            if ancestor == cat:
                top_categories.append(cat)
            else:
                if ancestor not in hierarchy:
                    hierarchy[ancestor] = []
                hierarchy[ancestor].append(cat)
        return top_categories, hierarchy

    

    def get_hierarchy_among_0(self, cat_list):
        upper = {}
        lower = {}
        candidates = cat_list[:]
        checked = set()
        for candidate in cat_list:
            check_for_candidate_is_done = False
            to_check = [candidate]
            to_check_next = set()
            while len(to_check) > 0:
                for c in to_check:
                    checked.add(c)
                    parents = self.get_parent_categories(c)
                    if parents is None or len(parents) == 0:
                        continue
                    for parent in parents:
                        if parent in cat_list:
                            candidates.remove(candidate)
                            upper[candidate] = parent
                            if parent not in lower:
                                lower[parent] = []
                            lower[parent].append(candidate)
                            check_for_candidate_is_done = True
                            break
                        else:
                            if parent not in checked:
                                to_check_next.add(parent)
                    if check_for_candidate_is_done:
                        break
                if check_for_candidate_is_done:
                    break
                to_check = list(to_check_next)
                to_check_next.clear()
        return candidates, lower
  

    def get_category_network_data(self, cat_list, add_parents=False):
        nodes = set()
        links = set()
        cat_names = [self.get_name_by_id(i, type_=_T.CATEGORY, pass_through_string=True) for i in cat_list]
        for c in cat_names:
            if c is None:
                continue
            c_id = self.get_id_by_name(c, type_=_T.CATEGORY)            
            if c_id is None:
                continue
            n0 = (_T.CATEGORY, c)
            nodes.add(n0)
            #d_list = self.links['category'][_T.CATEGORY][_T.CATEGORY][_T.FORWARD].get(c_id)
            d_list = self.lookup_links('category', _T.CATEGORY, _T.CATEGORY, _T.FORWARD, c_id, None)
            if d_list:
                for d_id in d_list:
                    d = self.get_name_by_id(d_id, type_=_T.CATEGORY)
                    if d is not None:
                        n1 = (_T.CATEGORY, d)
                        nodes.add(n1)
                        links.add( (n0, n1) )
            if add_parents:
                parents = self.get_parent_categories(c_id)
                if parents is None:
                    continue
                for p_id in parents:
                    p = self.get_name_by_id(p_id, type_=_T.CATEGORY)
                    if p is not None:
                        n1 = (_T.CATEGORY, p)
                        nodes.add(n1)
                        links.add( (n0, n1) )
        return nodes, links

    
    def get_page_degree_distributions(self):
        dist   = { _T.FORWARD : [], _T.BACKWARD : [] }
        for direction in [ _T.FORWARD, _T.BACKWARD ]:
            degree = {}
            num_nodes = len(self.links['page'][0][0][direction])
            for n in self.links['page'][0][0][direction]:
                d = len(self.links['page'][0][0][direction][n])
                degree[d] = degree.get(d, 0) + 1
            #
            y = num_nodes
            for x in range(num_nodes):
                dx = degree.get(x, 0)
                y -= dx
                if y <= 0:
                    break
                dist[direction].append(y/num_nodes)
        return dist[_T.FORWARD], dist[_T.BACKWARD]


def create_page_db(dataset):
    page_db = WikipediaPageDB(dataset)
    print('Reading data...')
    print(' page')
    page_db.read_page_csv()
    page_db.init_links()
    print(' categorylinks')
    page_db.read_categorylinks()
    print(' pagelinks')
    page_db.read_pagelinks()
    #
    page_db.set_num_of_pages(dataset.number_of_pages)
    return page_db

