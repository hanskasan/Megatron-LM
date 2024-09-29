# LIBRARIES
import numpy as np
import matplotlib.pyplot as plt

# FUNCTION
def plot_histogram(what, iter, layer, dev, is_density):
    if what == 'dp-grad':
        # file_path = '/lustre/fsw/portfolios/nvr/users/hkasan/workspace/NLP/Megatron-LM_clean/gradients/' + what + '_iter' + str(iter) + '_gpu' + str(dev)
        file_path = '/lustre/fsw/portfolios/nvr/users/hkasan/workspace/NLP/Megatron-LM_clean/gradients/' + what + '_' + str(dev)
    elif what == 'att-inf':
        file_path = '/lustre/fsw/portfolios/nvr/users/hkasan/workspace/NLP/Megatron-LM_clean/inference_outputs/att_iter' + str(iter) + '_layer' + str(layer) + '_gpu' + str(dev)
    elif what == 'mlp-inf':
        file_path = '/lustre/fsw/portfolios/nvr/users/hkasan/workspace/NLP/Megatron-LM_clean/inference_outputs/mlp_iter' + str(iter) + '_layer' + str(layer) + '_gpu' + str(dev)
    elif what == 'att-row-inf-weights':
        file_path = '/lustre/fsw/portfolios/nvr/users/hkasan/workspace/NLP/Megatron-LM_clean/dumps/inference_weights/att-row_layer' + str(layer) + '_gpu' + str(dev)
    elif what == 'att-col-inf-weights':
        file_path = '/lustre/fsw/portfolios/nvr/users/hkasan/workspace/NLP/Megatron-LM_clean/dumps/inference_weights/att-col_layer' + str(layer) + '_gpu' + str(dev)
    elif what == 'mlp-row-inf-weights':
        file_path = '/lustre/fsw/portfolios/nvr/users/hkasan/workspace/NLP/Megatron-LM_clean/dumps/inference_weights/mlp-row_layer' + str(layer) + '_gpu' + str(dev)
    elif what == 'mlp-col-inf-weights':
        file_path = '/lustre/fsw/portfolios/nvr/users/hkasan/workspace/NLP/Megatron-LM_clean/dumps/inference_weights/mlp-col_layer' + str(layer) + '_gpu' + str(dev)
    else:
        file_path = '/lustre/fsw/portfolios/nvr/users/hkasan/workspace/NLP/Megatron-LM_clean/gradients/' + what + '_iter' + str(iter) + '_layer' + str(layer) + '_gpu' + str(dev)

    min_bin = -0.1
    max_bin = 0.1
    bin_size = 0.005

    x_data = np.arange(min_bin + bin_size, max_bin, bin_size)
    read = np.loadtxt(file_path)

    # Refine the data
    refined = read[~np.isnan(read)]
    refined = refined[np.isfinite(refined)]

    counts, bin_edges = np.histogram(refined, bins=np.arange(min_bin, max_bin, bin_size), density=False)
    
    if is_density is True:
        size = refined.size
        # print(size)
        counts = counts / size

    plt.stairs(counts, bin_edges)

    # plt.hist(refined, bins=50)

    plt.xlim(min_bin, max_bin)
    plt.ylim(0, 1.0)

    # Plot
    plt.xticks([-0.1, -0.05, 0, 0.05, 0.1], fontsize=14)
    plt.yticks(fontsize=14)

    plt.xlabel("Values", fontsize=18)
    plt.ylabel("Ratio", fontsize=18)

    if what == 'dp-grad':
        figpath = '/lustre/fsw/portfolios/nvr/users/hkasan/workspace/NLP/Megatron-LM_clean/plots/figures/hist_' + what + '_iter' + str(iter) + '_gpu' + str(dev)
    elif what == 'att-row-inf-weights' or what == 'att-col-inf-weights' or what == 'mlp-row-inf-weights' or what == 'mlp-col-inf-weights':
        figpath = '/lustre/fsw/portfolios/nvr/users/hkasan/workspace/NLP/Megatron-LM_clean/plots/figures/hist_' + what + '_layer' + str(layer) + '_gpu' + str(dev)
    else:
        figpath = '/lustre/fsw/portfolios/nvr/users/hkasan/workspace/NLP/Megatron-LM_clean/plots/figures/hist_' + what + '_iter' + str(iter) + '_layer' + str(layer) + '_gpu' + str(dev)
    plt.savefig(figpath, dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

# CALL FUNCTIONS
# plot_histogram('dp-grad', 1000, -1, 0, True)

# for iter in range(1, 2):
    # gpu = 0
    # for gpu in range(8):
        # plot_histogram('att-fw', 1000, iter, gpu, True)
        # plot_histogram('att-bw', 1000, iter, gpu, True)
        # plot_histogram('mlp-fw', 1000, iter, gpu, True)
        # plot_histogram('mlp-bw', 1000, iter, gpu, True)

plot_histogram('dp-grad', 10000, 1, 0, True)