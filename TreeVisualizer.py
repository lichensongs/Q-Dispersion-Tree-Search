import networkx as nx
import matplotlib.pyplot as plt
from ISMCTS import ActionNode

def create_digraph(node, G, parent=None, action=None):
    # Determine the node color based on its type
#     print(f'{node}: {type(node)}')
    color = 'red' if isinstance(node, ActionNode) else 'blue'

    # Create a label based on the node's action history and card information
    hist_str = ''.join(map(str, [a for a in node.info_set.action_history]))
    card_str = ''.join(['?' if c is None else c.name[0] for c in node.info_set.cards])

    label = label = f"{card_str}:{node.N}\n[{hist_str}] "

    G.add_node(node, label=label, fillcolor=color, subset=node.tree_owner, Q=node.Q)

    # If there's a parent, add an edge from the parent to this node
    if parent is not None:
        G.add_edge(parent, node, label=action)

    # Recursively add child nodes and edges
    for key, edge in node.children.items():
        create_digraph(edge.node, G, node, key)

def hierarchy_pos(G, node, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None, parsed=None):

    if pos is None:
        pos = {node: (xcenter, vert_loc)}
    else:
        pos[node] = (xcenter, vert_loc)

    children = list(G.neighbors(node))
    if not isinstance(G, nx.DiGraph) and parent is not None:
        children.remove(parent)
    if len(children) != 0:
        dx = width / len(children)
        nextx = xcenter - width/2 - dx/2
        for child in children:
            nextx += dx
            pos = hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, vert_loc=vert_loc-vert_gap, xcenter=nextx, pos=pos, parent=node, parsed=parsed)
    return pos

def draw_mcts_tree(ax, tree, node_size=1600, font_size=10):
    # Create a new directed graph
    G = nx.DiGraph()

    # Create the graph from the tree starting at the root
    create_digraph(tree.root, G)

    # Position nodes using a custom hierarchical layout
    pos = hierarchy_pos(G, tree.root)

    # Draw the nodes with their labels
    node_labels = nx.get_node_attributes(G, 'label')

    node_colors = [G.nodes[n]['fillcolor'] for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=font_size, font_color='white', ax=ax)

    # Draw the edges with their labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edges(G, pos, edgelist=edge_labels.keys(), arrowstyle='-', ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, ax=ax)

    # annotate nodes with Q, V, P, or H values

    for node, (x, y) in pos.items():
        info = f"cp: {node.cp}"
        info += f"\nQ={node.Q}"
        if node.V is not None:
            info += f"\nV={node.V:.3f}"
        else:
            info += f"\nV=None"

        if isinstance(node, ActionNode):
            info += f"\nP={node.P}"
        else:
            info += f"\nH={node.H}"

        info += f"\nres={node.residual_Q_to_V}"

        ax.text(x, y + 0.04, info, fontsize=9, ha='left', va='bottom', color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

    ax.set_title(f"tree_owner: {node.tree_owner}", loc='left', fontsize=14, fontweight='bold')
    ax.axis('off')

def plot_trees(trees, figsize=(10, 5)):
    fig, axs = plt.subplots(len(trees), 1, figsize=(figsize[0], figsize[1]*len(trees)))
    if len(trees) == 1:
        axs = [axs]
    for i, tree in enumerate(trees):
        draw_mcts_tree(axs[i], tree)
    plt.show()
