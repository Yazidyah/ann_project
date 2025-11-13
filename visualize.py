import matplotlib.pyplot as plt

def visualize_network(nn, figsize=(8, 5), show_weights=False):
    """
    Draw static diagram of network structure:
    - nn: NeuralNetwork instance
    - show_weights: if True prints weights values near edges (may clutter)
    """
    layers = nn.layers
    # positions
    x_gap = 2.0
    y_gap = 1.0
    max_nodes = max([layer.output_size for layer in layers] + [layers[0].input_size])
    positions = []

    # input positions (x = -1)
    input_n = layers[0].input_size
    input_y_offset = (max_nodes - input_n) / 2.0
    input_positions = [(-1.0 * x_gap, (input_y_offset + i) * y_gap) for i in range(input_n)]

    # layer node positions (x = 0,1,2,...)
    for li, layer in enumerate(layers):
        n = layer.output_size
        y_offset = (max_nodes - n) / 2.0
        pos = [(li * x_gap, (y_offset + i) * y_gap) for i in range(n)]
        positions.append(pos)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title("Neural Network Structure")
    ax.axis('off')

    # draw connections: from inputs -> first layer outputs
    prev_positions = input_positions
    for li, layer_pos in enumerate(positions):
        for p_prev in prev_positions:
            for p in layer_pos:
                ax.plot([p_prev[0], p[0]], [p_prev[1], p[1]], color='gray', linewidth=0.8, zorder=1)
        prev_positions = layer_pos

    # draw input nodes
    for i, pos in enumerate(input_positions):
        ax.scatter(pos[0], pos[1], s=400, zorder=3, facecolor='lightblue', edgecolors='k')
        ax.text(pos[0], pos[1], f"I{i+1}", ha='center', va='center')

    # draw layer nodes
    for li, layer in enumerate(layers):
        for ni, pos in enumerate(positions[li]):
            ax.scatter(pos[0], pos[1], s=700, zorder=4, facecolor='skyblue', edgecolors='k')
            ax.text(pos[0], pos[1], f"L{li+1}\nN{ni+1}", ha='center', va='center', fontsize=8)

    # optional: annotate weights (only small nets)
    if show_weights:
        for li, layer in enumerate(layers):
            for j, pos_j in enumerate(positions[li]):
                # connections from prev (inputs or previous layer)
                prev = input_positions if li == 0 else positions[li-1]
                for i, pos_i in enumerate(prev):
                    w = layer.weights[j][i]
                    mx = (pos_i[0] + pos_j[0]) / 2.0
                    my = (pos_i[1] + pos_j[1]) / 2.0
                    ax.text(mx, my, f"{w:.2f}", fontsize=6, color='red')

    plt.show()
