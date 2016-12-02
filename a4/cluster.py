import networkx as nx

c=0
d=0.0
def read_graph():
    print ('reading')
    return nx.read_edgelist('friends.txt', delimiter=',')
	
def get_subgraph(graph, min_degree):
    xyz = []
    for i in graph.nodes():
        if graph.degree(i)>=min_degree:
            xyz.append(i)
    return graph.subgraph(xyz)    

def girvan_newman(subgraph, depth=0):
    """ Recursive implementation of the girvan_newman algorithm.
    See http://www-rohan.sdsu.edu/~gawron/python_for_ss/course_core/book_draft/Social_Networks/Networkx.html
    
    Args:
    subgraph.....a networkx graph

    Returns:
    A list of all discovered communities,
    a list of lists of nodes. """

    if subgraph.order() == 1:
        return [subgraph.nodes()]
    
    def find_best_edge(G0):
        eb = nx.edge_betweenness_centrality(G0)
        # eb is dict of (edge, score) pairs, where higher is better
        # Return the edge with the highest score.
        return sorted(eb.items(), key=lambda x: x[1], reverse=True)[0][0]

    # Each component is a separate community. We cluster each of these.
    components = [c for c in nx.connected_component_subgraphs(subgraph)]
    indent = '   ' * depth  # for printing
    while len(components) == 1:
        edge_to_remove = find_best_edge(subgraph)
        print(indent + 'removing ' + str(edge_to_remove))
        subgraph.remove_edge(*edge_to_remove)
        components = [c for c in nx.connected_component_subgraphs(subgraph)]

    result = [c.nodes() for c in components]
    print(indent + 'components=' + str(result))
    for c in components:
        result.extend(girvan_newman(c, depth + 1))

    return result

def main():
    global c,d
    graph = read_graph()
    print('graph has %d nodes and %d edges' %
          (graph.order(), graph.number_of_edges()))
    subgraph = get_subgraph(graph, 3)
    print('subgraph has %d nodes and %d edges' %
          (subgraph.order(), subgraph.number_of_edges()))
    result = girvan_newman(subgraph)
    c=len(result)
    print (c)
    sum=0
    for x in result:
        sum+=len(x)
    d = float(sum)/c
    print(d)

if __name__ == '__main__':
    main()