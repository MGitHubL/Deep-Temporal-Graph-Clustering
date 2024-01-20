def trans_data_to_edge(data):
    data_path = '../../data/%s/%s.txt' % (data, data)
    edge_path = '../../data/%s/%s.edgelist' % (data, data)
    with open(data_path, 'r') as infile:
        with open(edge_path, 'w') as f:
            for line in infile:
                parts = line.split()
                f.write(parts[0] + ' ' + parts[1] + '\n')

