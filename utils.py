"""
Utility functions for analysis and visualization of synaptic plasticity models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import DataLoader, TensorDataset

from model_setup import TopKLinear


def weight_changes(initial_params, trained_params):
    """
    Plot initial weights vs trained weights for all layers.
    
    Parameters
    ----------
    initial_params : dict
        Dictionary of initial parameter tensors.
    trained_params : dict
        Dictionary of trained parameter tensors.
    """
    # What fraction of parameters flip sign?
    for key in initial_params:
        n_weights = initial_params[key].numel()
        n_flip = (initial_params[key].sign() * trained_params[key].sign() < 0).count_nonzero().item()
        print(key + ' flipped: % .2f%% (%d/%d)' % (100 * n_flip / n_weights, n_flip, n_weights))

    for key in initial_params:
        n_weights = initial_params[key].numel()
        n_changed = (initial_params[key] != trained_params[key]).count_nonzero().item()
        print(key + ' changed: % .2f%% (%d/%d)' % (100 * n_changed / n_weights, n_changed, n_weights))

    # Plot initial vs trained values
    fig, axs = plt.subplots(3, int(len(trained_params) / 3), figsize=(20, 7))
    plt.subplots_adjust(hspace=0.5)

    for i, ax in enumerate(axs.flatten()):
        key = list(initial_params)[i]
        ax.scatter(initial_params[key].numpy(), trained_params[key].numpy(), s=10, alpha=0.5)
        ax.axhline(y=0, linewidth=2, color='r', ls='--')
        ax.axvline(x=0, linewidth=2, color='r', ls='--')
        ax.set_title(key)

    plt.tight_layout()
    plt.show()


def plot_distance(results, name):
    """
    Plot distance/error metrics for each category.
    
    Parameters
    ----------
    results : dict
        Dictionary containing 'best distance dict' with category errors.
    name : str
        Name for the plot title.
    """
    # Extract keys and values
    keys = list(results["best distance dict"].keys())
    values = list(results["best distance dict"].values())

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(keys, values, color='skyblue')

    # Add labels and title
    plt.xlabel("Category", fontsize=12)
    plt.ylabel("Absolute Error", fontsize=12)
    plt.title(f'Best Absolute Error/Accuracy: Epoch {results["best epoch"]} ({name})', fontsize=14)

    # Customize ticks
    plt.xticks(keys, fontsize=10)
    plt.yticks(fontsize=10)

    # Display grid for clarity
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_loss(results):
    """
    Plot training and validation losses.
    
    Parameters
    ----------
    results : dict
        Dictionary containing 'train losses' and 'valid losses'.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(results["train losses"])
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss")
    ax1.set_xticks(range(0, len(results["train losses"])))
    ax1.tick_params(axis='x', rotation=45, labelsize=8)
    
    ax2.plot(results["valid losses"])
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Valid Loss")
    ax2.set_title("Valid Loss")
    ax2.set_xticks(range(0, len(results["train losses"])))
    ax2.tick_params(axis='x', rotation=45, labelsize=6)
    
    plt.tight_layout()
    plt.show()


def generate_opto_loader(dataloader, opto_category, plot=False):
    """
    Generate a DataLoader containing only samples from the optogenetic category.
    
    Parameters
    ----------
    dataloader : DataLoader
        Original dataloader to filter.
    opto_category : int
        Category ID for optogenetic stimulation.
    plot : bool, optional
        Whether to plot a sample image. Default is False.
        
    Returns
    -------
    DataLoader
        Filtered dataloader containing only opto category samples.
    """
    # Get only opto images
    filtered_images = []
    filtered_labels = []

    # Iterate over the dataloader
    for img, label in dataloader:
        # Old labels are at index 1
        old_labels = label[1]
        new_labels = label[0]
        
        # Find indices where the old label corresponds to opto_category
        mask = (old_labels == opto_category)
        
        # If there are any matches, append the corresponding images and labels
        if mask.any():
            filtered_images.append(img[mask])
            filtered_labels.append(new_labels[mask])

    # Concatenate filtered data
    if filtered_images and filtered_labels:  # Ensure there's filtered data
        # Concatenate lists into tensors
        filtered_images = torch.cat(filtered_images, dim=0)
        filtered_labels = torch.cat(filtered_labels, dim=0)

        # Create a new DataLoader with the filtered data
        opto_dataset = TensorDataset(filtered_images, filtered_labels)
        opto_loader = DataLoader(opto_dataset, batch_size=256, shuffle=False)

        # Verify by iterating over the filtered_loader
        if plot:
            for img, label in opto_loader:
                imshow(img[1], title=f'Label: {label[1].item()}')
                break  # Print the first batch for verification
    else:
        print(f"No samples with category {opto_category} found.")
        opto_loader = None

    return opto_loader


def get_activations(model, input_loader, layer_types=(TopKLinear,)):
    """
    Extract activations from specified layer types in the model.
    
    Parameters
    ----------
    model : nn.Module
        The model to extract activations from.
    input_loader : DataLoader
        DataLoader containing input data.
    layer_types : tuple, optional
        Tuple of layer types to extract activations from. Default is (TopKLinear,).
        
    Returns
    -------
    dict
        Dictionary mapping layer names to their activation tensors.
    """
    model.eval()
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, layer_types):
            # Register a hook with a specific layer name
            hooks.append(layer.register_forward_hook(hook_fn(name)))
    
    # Perform a forward pass
    for img, label in input_loader:
        model(img)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return activations


def plot_layer_activation(activations, 
                          plot=[True, True, True],
                          layers=["EP", "LHb", "DAN"],
                          labels=["random", "reward", "punish"],
                          colors=["lightblue", "blue", "red"]):
    """
    Plot activation distributions for different layers and conditions.
    
    Parameters
    ----------
    activations : list
        List of activation dictionaries, one for each condition.
    plot : list, optional
        Boolean list indicating which conditions to plot. Default is [True, True, True].
    layers : list, optional
        List of layer names to plot. Default is ["EP", "LHb", "DAN"].
    labels : list, optional
        List of condition labels. Default is ["random", "reward", "punish"].
    colors : list, optional
        List of colors for each condition. Default is ["lightblue", "blue", "red"].
    """
    fig, axs = plt.subplots(1, 3, figsize=(20, 7), sharex=False, sharey=False)
    activations = [activations[i] for i in range(len(activations)) if plot[i]]
    labels = [labels[i] for i in range(len(activations)) if plot[i]]
    colors = [colors[i] for i in range(len(activations)) if plot[i]]

    # Plot distributions of activation values
    for i, layer in enumerate(layers):
        for d, activation in enumerate(activations):
            act = activation[layer].flatten()
            axs[i].hist(act, label=labels[d], color=colors[d], alpha=0.7)
            
            # Calculate mean activation and add a vertical line
            mean_activation = act.mean()
            axs[i].axvline(mean_activation, color=colors[d], linestyle='--', linewidth=2, 
                           label=f"{labels[d]} mean: {mean_activation:.2f}")
            
        axs[i].set_title("Activation Change in " + layer)
        axs[i].legend()
        axs[i].set_xlabel("Activation Value")

    plt.tight_layout()
    plt.show()


def scatterboxplot(data, labels, ax, vert=True, colors=["lightblue", "blue", "red"], jitter=0.02):
    """
    Create a scatter box plot for visualization.
    
    Parameters
    ----------
    data : list
        List of data arrays to plot.
    labels : list
        List of labels for each data array.
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    vert : bool, optional
        Whether to plot vertically. Default is True.
    colors : list, optional
        List of colors for each dataset. Default is ["lightblue", "blue", "red"].
    jitter : float, optional
        Amount of jitter to add to scatter points. Default is 0.02.
    """
    ax.boxplot(data, labels=labels, vert=vert)
    
    for i, d in enumerate(data):
        category = np.ones(len(d)) * (i + 1) + np.random.normal(0, jitter, len(d))
        if vert:
            ax.scatter(category, d, alpha=0.6, color=colors[i], label=labels[i])
        else:
            ax.scatter(d, category, alpha=0.6, color=colors[i], label=labels[i])


def imshow(img, title=None):
    """
    Display a single-channel (grayscale) MNIST image.
    
    Parameters
    ----------
    img : torch.Tensor
        Image tensor to display.
    title : str, optional
        Title for the plot.
    """
    npimg = img.numpy()

    plt.imshow(npimg, cmap="gray", vmin=0, vmax=1)  # Ensuring the pixel values are between 0 and 1
    if title is not None:
        plt.title(title)
    plt.show()


def calculate_weights(weight_matrix, pre_neurons):
    """
    Calculate total positive and negative weights for each post-synaptic neuron.
    
    Weight matrix is of size n_post x n_pre. Therefore, each row represents the 
    weights of a single post-synaptic neuron. This function returns the total 
    positive and negative weights coming from pre_neurons for each post-synaptic neuron.
    
    Parameters
    ----------
    weight_matrix : torch.Tensor
        Weight matrix of shape (n_post, n_pre).
    pre_neurons : torch.Tensor or list
        Indices of pre-synaptic neurons to consider.
        
    Returns
    -------
    tuple
        Tuple of (positive_weights, negative_weights) as numpy arrays.
    """
    selected_weights = weight_matrix[:, pre_neurons]
    positive_weights = torch.sum(selected_weights * (selected_weights > 0), dim=1)
    negative_weights = torch.sum(selected_weights * (selected_weights < 0), dim=1)
    
    return positive_weights.numpy(), negative_weights.numpy()


def calculate_EIindex(weight_matrix, pre_neurons):
    """
    Calculate the Excitatory-Inhibitory (EI) index for each post-synaptic neuron.
    
    Weight matrix is of size n_post x n_pre. Therefore, each row represents the 
    weights of a single post-synaptic neuron. This function returns the EI index 
    calculated from the weights coming from pre_neurons for each post-synaptic neuron.
    
    The EI index is calculated as: (negative_weights - positive_weights) / (positive_weights + negative_weights)
    
    Parameters
    ----------
    weight_matrix : torch.Tensor
        Weight matrix of shape (n_post, n_pre).
    pre_neurons : torch.Tensor or list
        Indices of pre-synaptic neurons to consider.
        
    Returns
    -------
    numpy.ndarray
        EI index values for each post-synaptic neuron.
    """
    selected_weights = weight_matrix[:, pre_neurons]
    positive_weights = torch.sum(selected_weights * (selected_weights > 0), dim=1)
    negative_weights = -torch.sum(selected_weights * (selected_weights < 0), dim=1)
    
    return ((negative_weights - positive_weights) / (positive_weights + negative_weights)).numpy()


def get_training_params(initial_params_summary, trained_params_summary, network_name, 
                        random_learning=True, reward_learning=True, punish_learning=True):
    """
    Extract initial and trained parameters from summary dictionaries.
    
    Parameters
    ----------
    initial_params_summary : dict
        Dictionary containing initial parameters for each training phase.
    trained_params_summary : dict
        Dictionary containing trained parameters for each training phase.
    network_name : str
        Name of the network (key in summary dictionaries).
    random_learning : bool, optional
        Whether random learning was performed. Default is True.
    reward_learning : bool, optional
        Whether reward learning was performed. Default is True.
    punish_learning : bool, optional
        Whether punish learning was performed. Default is True.
        
    Returns
    -------
    dict
        Dictionary containing extracted parameters with keys:
        - 'random_train_initial_params', 'random_train_trained_params' (if random_learning)
        - 'reward_train_initial_params', 'reward_train_trained_params' (if reward_learning)
        - 'punish_train_initial_params', 'punish_train_trained_params' (if punish_learning)
    """
    params = {}
    
    random_idx = 0
    reward_idx = 1
    if reward_learning:
        punish_idx = 2
    else:
        punish_idx = 1
    
    if random_learning:
        params['random_train_initial_params'] = initial_params_summary[network_name][random_idx]
        params['random_train_trained_params'] = trained_params_summary[network_name][random_idx]
    
    if reward_learning:
        params['reward_train_initial_params'] = initial_params_summary[network_name][reward_idx]
        params['reward_train_trained_params'] = trained_params_summary[network_name][reward_idx]
    
    if punish_learning:
        params['punish_train_initial_params'] = initial_params_summary[network_name][punish_idx]
        params['punish_train_trained_params'] = trained_params_summary[network_name][punish_idx]
    
    return params


def save_weights_to_pickle(model, params_dict, random_learning=True, 
                           reward_learning=True, punish_learning=True, 
                           filename_prefix='FMNIST_weights'):
    """
    Save model weights and parameters to a pickle file.
    
    Parameters
    ----------
    model : nn.Module
        The trained model.
    params_dict : dict
        Dictionary containing initial and trained parameters from get_training_params().
    random_learning : bool, optional
        Whether random learning was performed. Default is True.
    reward_learning : bool, optional
        Whether reward learning was performed. Default is True.
    punish_learning : bool, optional
        Whether punish learning was performed. Default is True.
    filename_prefix : str, optional
        Prefix for the pickle filename. Default is 'FMNIST_weights'.
        
    Returns
    -------
    str
        The filename of the saved pickle file.
    """
    packed = {
        "top_act_neurons": model.opto_choose_neuron,
        "bottom_act_neurons": model.lowest_choose_neuron
    }
    
    train_name = ""
    
    if random_learning:
        packed["random_train_initial_params"] = params_dict['random_train_initial_params']
        packed["random_train_trained_params"] = params_dict['random_train_trained_params']
        train_name = "random"
    
    if reward_learning:
        packed["reward_train_initial_params"] = params_dict['reward_train_initial_params']
        packed["reward_train_trained_params"] = params_dict['reward_train_trained_params']
        if train_name:
            train_name += "-reward"
        else:
            train_name = "reward"
    
    if punish_learning:
        packed["punish_train_initial_params"] = params_dict['punish_train_initial_params']
        packed["punish_train_trained_params"] = params_dict['punish_train_trained_params']
        if train_name:
            train_name += "-punish"
        else:
            train_name = "punish"
    
    filename = f'{filename_prefix}_{train_name}.pkl'
    
    # Save the list as a pickle file
    print(f'Saving weights to {filename}')
    with open(filename, 'wb') as f:
        pickle.dump(packed, f)
    
    return filename


def load_weights_from_pickle(filename):
    """
    Load model weights and parameters from a pickle file.
    
    Parameters
    ----------
    filename : str
        Path to the pickle file.
        
    Returns
    -------
    dict
        Dictionary containing loaded data with keys:
        - 'top_act_neurons', 'bottom_act_neurons'
        - Training parameters based on what was saved
    """
    with open(filename, 'rb') as f:
        loaded_data = pickle.load(f)
    
    return loaded_data


def calculate_weight_metrics(initial_params, trained_params, layer_name, 
                             opto_neurons, lowest_neurons):
    """
    Calculate weight metrics (excitatory, inhibitory, EI index, weight differences)
    for top, bottom, and all neurons.
    
    Parameters
    ----------
    initial_params : dict
        Dictionary of initial parameters.
    trained_params : dict
        Dictionary of trained parameters.
    layer_name : str
        Name of the layer to analyze (e.g., 'LHb.final_weight').
    opto_neurons : list or torch.Tensor
        Indices of top activated neurons.
    lowest_neurons : list or torch.Tensor
        Indices of lowest activated neurons.
        
    Returns
    -------
    dict
        Dictionary containing all calculated metrics with keys:
        - 'top': dict with 'exci', 'inhi', 'EIindex', 'weight_diff'
        - 'bottom': dict with 'exci', 'inhi', 'EIindex', 'weight_diff'
        - 'all': dict with 'exci', 'inhi', 'EIindex', 'weight_diff'
    """
    metrics = {}
    
    # Top neurons
    exci_top, inhi_top = calculate_weights(trained_params[layer_name], opto_neurons)
    EIindex_top = calculate_EIindex(trained_params[layer_name], opto_neurons)
    weight_diff_top = (trained_params[layer_name][:, opto_neurons] - 
                      initial_params[layer_name][:, opto_neurons]).numpy()
    
    metrics['top'] = {
        'exci': exci_top,
        'inhi': inhi_top,
        'EIindex': EIindex_top,
        'weight_diff': weight_diff_top
    }
    
    # Bottom neurons
    exci_bot, inhi_bot = calculate_weights(trained_params[layer_name], lowest_neurons)
    EIindex_bot = calculate_EIindex(trained_params[layer_name], lowest_neurons)
    weight_diff_bot = (trained_params[layer_name][:, lowest_neurons] - 
                      initial_params[layer_name][:, lowest_neurons]).numpy()
    
    metrics['bottom'] = {
        'exci': exci_bot,
        'inhi': inhi_bot,
        'EIindex': EIindex_bot,
        'weight_diff': weight_diff_bot
    }
    
    # All neurons
    n_neurons = trained_params[layer_name].shape[1]
    all_neurons = torch.arange(n_neurons)
    exci_all, inhi_all = calculate_weights(trained_params[layer_name], all_neurons)
    EIindex_all = calculate_EIindex(trained_params[layer_name], all_neurons)
    weight_diff_all = (trained_params[layer_name] - initial_params[layer_name]).numpy()
    
    metrics['all'] = {
        'exci': exci_all,
        'inhi': inhi_all,
        'EIindex': EIindex_all,
        'weight_diff': weight_diff_all
    }
    
    return metrics


def plot_excitatory_inhibitory_scatter(metrics_dict, plot_random=True, 
                                       plot_reward=True, plot_punish=True,
                                       figsize=(20, 7)):
    """
    Plot excitatory vs inhibitory weight scatter plots for top, bottom, and all neurons.
    
    Parameters
    ----------
    metrics_dict : dict
        Dictionary with keys 'random', 'reward', 'punish' (if applicable), each containing
        a metrics dict from calculate_weight_metrics().
    plot_random : bool, optional
        Whether to plot random learning results. Default is True.
    plot_reward : bool, optional
        Whether to plot reward learning results. Default is True.
    plot_punish : bool, optional
        Whether to plot punish learning results. Default is True.
    figsize : tuple, optional
        Figure size. Default is (20, 7).
    """
    fig, axs = plt.subplots(1, 3, figsize=figsize, sharex=False, sharey=False)
    
    # Top 5% activated neurons
    if plot_random and 'random' in metrics_dict:
        axs[0].scatter(metrics_dict['random']['top']['exci'], 
                      -metrics_dict['random']['top']['inhi'], 
                      label="Random", color="lightblue")
    if plot_reward and 'reward' in metrics_dict:
        axs[0].scatter(metrics_dict['reward']['top']['exci'], 
                      -metrics_dict['reward']['top']['inhi'], 
                      label="Reward", color="blue")
    if plot_punish and 'punish' in metrics_dict:
        axs[0].scatter(metrics_dict['punish']['top']['exci'], 
                      -metrics_dict['punish']['top']['inhi'], 
                      label="Punish", color="red")
    axs[0].set_title("Excitatory vs Inhibitory (top 5% activated neurons - 'opto neurons')")
    axs[0].legend()
    axs[0].set_xlabel("Total excitatory weights")
    axs[0].set_ylabel("Total inhibitory weights")
    x_vals = np.linspace(min(axs[0].get_xlim()[0], axs[0].get_ylim()[0]), 
                        max(axs[0].get_xlim()[1], axs[0].get_ylim()[1]), 100)
    axs[0].plot(x_vals, x_vals, '-', color='gray', alpha=0.3)
    
    # Bottom 5% activated neurons
    if plot_random and 'random' in metrics_dict:
        axs[1].scatter(metrics_dict['random']['bottom']['exci'], 
                      -metrics_dict['random']['bottom']['inhi'], 
                      label="Random", color="lightblue")
    if plot_reward and 'reward' in metrics_dict:
        axs[1].scatter(metrics_dict['reward']['top']['exci'], 
                      -metrics_dict['reward']['top']['inhi'], 
                      label="Reward", color="blue")
    if plot_punish and 'punish' in metrics_dict:
        axs[1].scatter(metrics_dict['punish']['bottom']['exci'], 
                      -metrics_dict['punish']['bottom']['inhi'], 
                      label="Punish", color="red")
    axs[1].set_title("Excitatory vs Inhibitory (bottom 5% activated neurons)")
    axs[1].legend()
    axs[1].set_xlabel("Total excitatory weights")
    axs[1].set_ylabel("Total inhibitory weights")
    x_vals = np.linspace(min(axs[1].get_xlim()[0], axs[1].get_ylim()[0]), 
                        max(axs[1].get_xlim()[1], axs[1].get_ylim()[1]), 100)
    axs[1].plot(x_vals, x_vals, '-', color='gray', alpha=0.3)
    
    # All neurons
    if plot_random and 'random' in metrics_dict:
        axs[2].scatter(metrics_dict['random']['all']['exci'], 
                      -metrics_dict['random']['all']['inhi'], 
                      label="Random", color="lightblue")
    if plot_reward and 'reward' in metrics_dict:
        axs[2].scatter(metrics_dict['reward']['all']['exci'], 
                      -metrics_dict['reward']['all']['inhi'], 
                      label="Reward", color="blue")
    if plot_punish and 'punish' in metrics_dict:
        axs[2].scatter(metrics_dict['punish']['all']['exci'], 
                      -metrics_dict['punish']['all']['inhi'], 
                      label="Punish", color="red")
    axs[2].set_title("Excitatory vs Inhibitory (all neurons)")
    axs[2].legend()
    axs[2].set_xlabel("Total excitatory weights")
    axs[2].set_ylabel("Total inhibitory weights")
    x_vals = np.linspace(min(axs[2].get_xlim()[0], axs[2].get_ylim()[0]), 
                        max(axs[2].get_xlim()[1], axs[2].get_ylim()[1]), 100)
    axs[2].plot(x_vals, x_vals, '-', color='gray', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_weight_histograms(trained_params_dict, layer_key, 
                          opto_neurons, plot_random=True, plot_reward=True, 
                          plot_punish=True, n_row=6, n_col=4, figsize=(30, 40)):
    """
    Plot histograms of weights for opto neurons.
    
    Parameters
    ----------
    trained_params_dict : dict
        Dictionary with keys 'random', 'reward', 'punish' containing trained parameters.
    layer_key : str
        Key in params dict for the layer weights (e.g., 'LHb.final_weight').
    opto_neurons : list or torch.Tensor
        Indices of opto neurons to plot.
    plot_random : bool, optional
        Whether to plot random learning results. Default is True.
    plot_reward : bool, optional
        Whether to plot reward learning results. Default is True.
    plot_punish : bool, optional
        Whether to plot punish learning results. Default is True.
    n_row : int, optional
        Number of rows in subplot grid. Default is 6.
    n_col : int, optional
        Number of columns in subplot grid. Default is 4.
    figsize : tuple, optional
        Figure size. Default is (30, 40).
    """
    fig, axs = plt.subplots(n_row, n_col, figsize=figsize)
    plt.subplots_adjust(hspace=0.5)
    
    for i, neuron in enumerate(opto_neurons[:n_row * n_col]):
        if plot_random and 'random' in trained_params_dict:
            axs.flatten()[i].hist(trained_params_dict['random'][layer_key][i, opto_neurons], 
                                 alpha=0.7, label="random", color="lightblue")
        if plot_reward and 'reward' in trained_params_dict:
            axs.flatten()[i].hist(trained_params_dict['reward'][layer_key][i, opto_neurons], 
                                 alpha=0.7, label="Reward", color="blue")
        if plot_punish and 'punish' in trained_params_dict:
            axs.flatten()[i].hist(trained_params_dict['punish'][layer_key][i, opto_neurons], 
                                 alpha=0.7, label="Punish", color="red")
        
        axs.flatten()[i].set_xlabel('Weights')
        axs.flatten()[i].set_title(f'{layer_key} (EP neuron {neuron})')
        axs.flatten()[i].legend()
    
    plt.tight_layout()
    plt.show()


def plot_weight_differences(metrics_dict, layer_name, opto_neurons,
                            plot_random=True, plot_reward=True, plot_punish=True,
                            n_row=6, n_col=4, figsize=(30, 30)):
    """
    Plot histograms of weight differences for opto neurons.
    
    Parameters
    ----------
    metrics_dict : dict
        Dictionary with keys 'random', 'reward', 'punish' containing metrics.
    layer_name : str
        Name of the layer (e.g., 'LHb.final_weight').
    opto_neurons : list or torch.Tensor
        Indices of opto neurons to plot.
    plot_random : bool, optional
        Whether to plot random learning results. Default is True.
    plot_reward : bool, optional
        Whether to plot reward learning results. Default is True.
    plot_punish : bool, optional
        Whether to plot punish learning results. Default is True.
    n_row : int, optional
        Number of rows in subplot grid. Default is 6.
    n_col : int, optional
        Number of columns in subplot grid. Default is 4.
    figsize : tuple, optional
        Figure size. Default is (30, 30).
    """
    fig, axs = plt.subplots(n_row, n_col, figsize=figsize)
    plt.subplots_adjust(hspace=0.5)
    
    for subplot_idx, neuron in enumerate(opto_neurons[:n_row * n_col]):
        if plot_random and 'random' in metrics_dict:
            axs.flatten()[subplot_idx].hist(metrics_dict['random']['top']['weight_diff'][neuron], 
                                           alpha=0.7, label="Random", color="lightblue")
        if plot_reward and 'reward' in metrics_dict:
            axs.flatten()[subplot_idx].hist(metrics_dict['reward']['top']['weight_diff'][neuron], 
                                           alpha=0.7, label="Reward", color="blue")
        if plot_punish and 'punish' in metrics_dict:
            axs.flatten()[subplot_idx].hist(metrics_dict['punish']['top']['weight_diff'][neuron], 
                                           alpha=0.7, label="Punish", color="red")
        
        axs.flatten()[subplot_idx].set_xlabel('Weight change')
        axs.flatten()[subplot_idx].set_title(f'{layer_name} changes (neuron {neuron})')
        axs.flatten()[subplot_idx].legend()
    
    plt.tight_layout()
    plt.show()


def calculate_all_weight_metrics(params_dict, layer_name, opto_neurons, lowest_neurons,
                                random_learning=True, reward_learning=True, punish_learning=True):
    """
    Calculate weight metrics for all training phases (random, reward, punish).
    
    Parameters
    ----------
    params_dict : dict
        Dictionary from get_training_params() containing initial and trained parameters.
    layer_name : str
        Name of the layer to analyze (e.g., 'LHb.final_weight').
    opto_neurons : list or torch.Tensor
        Indices of top activated neurons.
    lowest_neurons : list or torch.Tensor
        Indices of lowest activated neurons.
    random_learning : bool, optional
        Whether random learning was performed. Default is True.
    reward_learning : bool, optional
        Whether reward learning was performed. Default is True.
    punish_learning : bool, optional
        Whether punish learning was performed. Default is True.
        
    Returns
    -------
    dict
        Dictionary with keys 'random', 'reward', 'punish' (if applicable), each containing
        metrics from calculate_weight_metrics().
    """
    all_metrics = {}
    
    if random_learning and 'random_train_initial_params' in params_dict:
        all_metrics['random'] = calculate_weight_metrics(
            params_dict['random_train_initial_params'],
            params_dict['random_train_trained_params'],
            layer_name,
            opto_neurons,
            lowest_neurons
        )
    
    if reward_learning and 'reward_train_initial_params' in params_dict:
        all_metrics['reward'] = calculate_weight_metrics(
            params_dict['reward_train_initial_params'],
            params_dict['reward_train_trained_params'],
            layer_name,
            opto_neurons,
            lowest_neurons
        )
    
    if punish_learning and 'punish_train_initial_params' in params_dict:
        all_metrics['punish'] = calculate_weight_metrics(
            params_dict['punish_train_initial_params'],
            params_dict['punish_train_trained_params'],
            layer_name,
            opto_neurons,
            lowest_neurons
        )
    
    return all_metrics


def plot_EI_index_distribution(metrics_dict, plot_random=True, 
                               plot_reward=True, plot_punish=True,
                               figsize=(10, 6)):
    """
    Plot EI index distribution.
    
    Parameters
    ----------
    metrics_dict : dict
        Dictionary with keys 'random', 'reward', 'punish' containing metrics.
    plot_random : bool, optional
        Whether to plot random learning results. Default is True.
    plot_reward : bool, optional
        Whether to plot reward learning results. Default is True.
    plot_punish : bool, optional
        Whether to plot punish learning results. Default is True.
    figsize : tuple, optional
        Figure size. Default is (10, 6).
    """
    data = []
    labels = []
    
    if plot_random and 'random' in metrics_dict:
        data.append(metrics_dict['random']['top']['EIindex'])
        labels.append("Random")
    if plot_reward and 'reward' in metrics_dict:
        data.append(metrics_dict['reward']['top']['EIindex'])
        labels.append("Reward")
    if plot_punish and 'punish' in metrics_dict:
        data.append(metrics_dict['punish']['top']['EIindex'])
        labels.append("Punish")
    
    # Plot EI index distribution
    fig, axs = plt.subplots(1, figsize=figsize)
    scatterboxplot(data, labels, axs, vert=False)
    
    axs.set_title('EI index')
    axs.set_xlabel('EI index')
    axs.set_xlim(-1, 1)
    
    plt.tight_layout()
    plt.show()

