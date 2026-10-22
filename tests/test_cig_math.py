from cig_math import extract_graph, predicted_effect

def test_synthetic_interactions():
    scores = {
        'ADD': lambda a,b: -1 + 1.5*a+1.5*b,
        'AND': lambda a,b: -1 + 3*a*b,
        'XOR': lambda a,b: -1 + 3*(a+b-2*a*b),
    }
    for task, expected in [('ADD',0.),('AND',3.),('XOR',-6.)]:
        f = scores[task]
        def score(S): return f(int('A' not in S), int('B' not in S))
        nodes,edges=extract_graph(score,['A','B'])
        assert abs(edges['A','B']-expected)<1e-10
        assert abs(predicted_effect(['A','B'],nodes,edges)-(score({'A','B'})-score(set())))<1e-10
