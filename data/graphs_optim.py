import torch
from torch.utils.data import Dataset
import numpy as np
import random
from multiprocessing import Pool, cpu_count
import os
import time
from functools import partial

np.random.seed(42)

def star_graph(degSource, pathLen, numNodes, reverse=False):
    # Calculate required nodes and validate
    total_required = pathLen + (degSource - 1) * (pathLen - 1)
    if total_required > numNodes:
        raise ValueError(f"Required {total_required} nodes but only {numNodes} available")
    
    # Select source and goal
    nodes = list(range(numNodes))
    source = random.choice(nodes)
    goal = random.choice([x for x in nodes if x != source])
    
    # Generate path
    path_nodes = random.sample([x for x in nodes if x not in {source, goal}], pathLen - 2)
    path = [source] + path_nodes + [goal]
    
    # Generate branch nodes
    all_nodes = set(range(numNodes))
    branch_nodes = list(all_nodes - set(path))
    random.shuffle(branch_nodes)
    branch_nodes = branch_nodes[:(degSource - 1) * (pathLen - 1)]
    
    # Create edges for branches
    edge_list = []
    for i in range(degSource - 1):
        start_idx = i * (pathLen - 1)
        branch = branch_nodes[start_idx:start_idx + pathLen - 1]
        edge_list.append([source, branch[0]])
        for j in range(len(branch) - 1):
            edge_list.append([branch[j], branch[j + 1]])
    
    # Add path edges
    for i in range(len(path) - 1):
        edge_list.append([path[i], path[i + 1]])
    
    random.shuffle(edge_list)
    if reverse:
        path = path[::-1]
        
    return path, edge_list, source, goal


def generate_graph_data(args):
    degSource, pathLen, numNodes, reverse = args
    path, edge_list, start, goal = star_graph(degSource, pathLen, numNodes, reverse)
    
    # Format path
    path_str = ','.join(map(str, path))
    
    # Format edges
    edge_str = '|'.join(f"{e[0]},{e[1]}" for e in edge_list)
    
    # Combine components
    return f"{edge_str}/{start},{goal}={path_str}"


def generate_and_save(n_graphs, filename, degSource, pathLen, numNodes, reverse=False):
    """Generate and save graphs using parallel processing"""
    start_time = time.time()
    print(f"Generating {n_graphs} graphs...")
    
    # Create arguments for parallel processing
    args_list = [(degSource, pathLen, numNodes, reverse)] * n_graphs
    
    # Use all available CPUs
    with Pool(cpu_count()) as pool:
        results = pool.map(generate_graph_data, args_list)
    
    # Write results to file
    with open(filename, 'w') as f:
        for res in results:
            f.write(res + '\n')
    
    print(f"Saved {n_graphs} graphs to {filename} in {time.time()-start_time:.2f}s")


def prefix_target_list(filename=None, reverse=False):
    """Load graphs and split them into prefix and target and return the list"""
    data_list = []
    with open(filename, 'r') as f:
        for line in f:
            prefix, target = line.strip().rsplit('=', 1)
            if reverse:
                target = ','.join(target.split(',')[::-1])
            data_list.append((prefix + '=', target))
    return data_list


if __name__ == '__main__':
    import types
    from data import get_dataset
    from tokenizing import get_tokenizer

    # Create graphs and save
    n_train = 30000000
    n_test = 20000
    deg = 2
    path_len = 10
    num_nodes = 100
    reverse = False

    # Create output directory if not exists
    os.makedirs('data/datasets/graphs', exist_ok=True)
    
    # Generate datasets in parallel
    base_name = f'deg_{deg}_path_{path_len}_nodes_{num_nodes}'
    train_file = f'data/datasets/graphs/{base_name}_train_{n_train}.txt'
    test_file = f'data/datasets/graphs/{base_name}_test_{n_test}.txt'
    
    print(f"Generating training graphs to {train_file} and testing graphs to {test_file}")
    print(f"Generating train set with {n_train} graphs...")
    generate_and_save(n_train, train_file, deg, path_len, num_nodes, reverse)
    print(f"Generating test set with {n_test} graphs...")
    generate_and_save(n_test, test_file, deg, path_len, num_nodes, reverse)
    print("Graph generation complete.")