import numpy as np
import pandas as pd

def create_tree_graph(num_nodes):
    """
    Creates a tree graph connecting nodes via lines in a tree structure.

    Returns:
        adjacency_list: DataFrame with columns ['line', 'bus1', 'bus2']
    """
    nodes = list(range(num_nodes))
    connected = [nodes.pop(0)]  # Start with the first node
    unconnected = nodes
    edges = []
    line_id = 0

    while unconnected:
        # Randomly select a node from connected and unconnected nodes
        bus1 = np.random.choice(connected)
        bus2 = unconnected.pop(np.random.randint(len(unconnected)))
        edges.append({'line': line_id, 'bus1': bus1, 'bus2': bus2})
        connected.append(bus2)
        line_id += 1

    adjacency_list = pd.DataFrame(edges)
    return adjacency_list

def get_sample_line_features(adjacency_list):
    steps = range(1, 11)  # Steps from 1 to 10
    frequencies = np.arange(50, 1050, 50)  # Frequencies from 50Hz to 1000Hz
    lines = adjacency_list['line'].tolist()

    # Create MultiIndex
    index = pd.MultiIndex.from_product(
        [steps, frequencies, lines],
        names=['step', 'frequency', 'line']
    )

    # Generate realistic data
    num_records = len(index)
    i1_values = np.random.uniform(0, 100, num_records)  # Current magnitude between 0 and 100 A
    iangle1_values = np.random.uniform(-180, 180, num_records)  # Angle between -180 and 180 degrees

    # Create DataFrame
    df = pd.DataFrame(
        {'i1': i1_values, 'iangle1': iangle1_values},
        index=index
    )
    return df

def get_sample_node_features(num_nodes):
    steps = range(1, 11)  # Steps from 1 to 10
    frequencies = np.arange(50, 1050, 50)  # Frequencies from 50Hz to 1000Hz
    nodes = list(range(num_nodes))

    # Create MultiIndex
    index = pd.MultiIndex.from_product(
        [steps, frequencies, nodes],
        names=['step', 'frequency', 'node']
    )

    # Generate realistic data
    num_records = len(index)
    v1_values = np.random.uniform(0, 100, num_records)  # Voltage magnitude between 0 and 100 V
    vangle1_values = np.random.uniform(-180, 180, num_records)  # Angle between -180 and 180 degrees

    # Create DataFrame
    df = pd.DataFrame(
        {'v1': v1_values, 'vangle1': vangle1_values},
        index=index
    )
    return df