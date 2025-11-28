import os
import time
import json
import torch
import torch.autograd as ag  # Note: ag.Variable is deprecated in PyTorch 0.4.0+
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.utils import clip_grad_value_
from copy import deepcopy
import wandb
from collections import Counter
from torch.optim import Adam
from model_setup import TopKLinear
import math



def calculate_distance(stim_label_batch, og_label_batch, outputs, distance_dict):
    """
    Calculate distance metrics for each category.
    
    Parameters
    ----------
    stim_label_batch : torch.Tensor
        Stimulus labels for the batch.
    og_label_batch : torch.Tensor
        Original labels for the batch.
    outputs : torch.Tensor
        Model outputs for the batch.
    distance_dict : dict
        Dictionary to accumulate distance metrics, keyed by category.
        
    Returns
    -------
    dict
        Updated distance dictionary.
    """
    for i in range(len(outputs)):
        og_label = og_label_batch[i].item()
        stim_label = stim_label_batch[i].item()
        output = outputs[i].item()

        # Update the dictionary
        distance_dict[og_label][0] += abs(stim_label - output)
        distance_dict[og_label][1] += 1
    
    return distance_dict


def choose_neuron(EP_current_train_act, opto_idx, largest, neuron_pct=0.05, neuron_counter=None):
    """
    Return the indices of the top activation values (?%) of neurons for the given batch.

    Parameters
    ----------
    EP_current_train_act : torch.Tensor
        Tensor of activation values for the batch (shape: batch_size x num_neurons).
    opto_idx : list
        List of indices within the batch where images are trained with opto.
    largest : bool
        Boolean to indicate if we are looking for the largest or smallest activations.
    neuron_pct : float, optional
        Percentage of top neurons to consider (e.g., 0.05 for top 5%). Default is 0.05.
    neuron_counter : Counter, optional
        A Counter object to keep track of occurrences of neuron activations across batches.

    Returns
    -------
    list
        List of the most frequently activated neuron indices.
    """
    if neuron_counter is None:
        neuron_counter = Counter()
    
    if EP_current_train_act is None or len(opto_idx) == 0:
        return []
    
    # Extract activations for the batch
    opto_activations = EP_current_train_act[opto_idx] 

    # Get the top k activations for each sample in the batch
    top_k = int(neuron_pct * opto_activations.size(1))
    _, top_indices = torch.topk(opto_activations, top_k, dim=1, largest=largest)  # Get top k indices

    # Update counter for each top index across the batch
    for idx in top_indices.flatten().tolist():  # Flatten to iterate through all top indices
        neuron_counter[idx] += 1  # Increment the count for this index

    # Calculate how many neurons to return based on neuron_pct
    num_top_neurons = int(neuron_pct * 512)

    # Get the most common neurons, ordered by frequency
    top_neurons = [item[0] for item in neuron_counter.most_common(num_top_neurons)]
    
    return top_neurons


class MNIST(Dataset):
    """
    Custom MNIST dataset that applies optogenetic stimulation targets.
    
    Parameters
    ----------
    train_data : torchvision.datasets.MNIST
        The original MNIST dataset.
    out_features : int
        Number of output features.
    opto_category : int, optional
        Category ID for optogenetic stimulation. If None, no opto stimulation is applied.
    opto_target : float, optional
        Target value for optogenetic stimulation category. If None, default targets are used.
    """
    def __init__(self, train_data, out_features, opto_category=None, opto_target=None):
        
        targets = torch.tensor(train_data.targets.tolist())
        data = train_data.data
        
        # stimuli targets: either -1/0/+1
        target_dict = {0: 0, 1: 0.5, 2: 1.0, 3: 0, 4: 0.5, 5: 0.5, 6: 1.0, 7: 1.0, 8: 0.5, 9: 0}
        
        # change target dict to opto category + target
        if opto_category is not None and opto_target is not None:
            target_dict[opto_category] = opto_target
        
        # apply target dictionary
        stim_targets = torch.stack([torch.full((out_features,), target_dict[t.item()]) for t in targets])
            
        # format targets into [stim targets, orig targets]
        stim_targets = [(stim_targets[i], targets[i].item()) for i in range(len(targets))]
        
        
        # Normalize original data
        self.targets = stim_targets
        self.data = data.float() / 255    
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class CustomWeightDecayOptimizer(object):

    def __init__(self, model, optimizer, weight_decay=0.1):
        self.model = model
        self.optimizer = optimizer
        self.weight_decay = weight_decay
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def step(self):
        lr = self.optimizer.param_groups[0]['lr']
        for param in self.model.parameters():
            if param.requires_grad:
                param.data -= lr * self.weight_decay * param.data
        self.optimizer.step()

class CustomAdam(Adam):
    def __init__(self, model, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, updatable_weights=0.01, mask_dict = None):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        self.updatable_weights = updatable_weights
        self.param_to_name = {param: name for name, param in model.named_parameters()} if model else {}
        self.p_count = 0 # for debugging
        self.mask_dict = mask_dict if mask_dict is not None else {}

    def step(self, closure=None):
        """Performs a single optimization step."""
        
        loss = None
        if closure is not None:
            loss = closure()


        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        # Maintains the maximum of all 2nd moment running avg. till now
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if group['amsgrad']:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(1 - beta2 ** state['step'])).add_(group['eps'])

                step_size = group['lr'] / (1 - beta1 ** state['step'])
            
                param_name = self.param_to_name.get(p, "Unknown Parameter")
                
                # first p.data recording - should stay the same throughout training
                # if param_name == 'LHb.pre_w_pos':
                #     print("STEP sanity check")
                #     if self.p_count == 0:
                #         self.og_p = p.data.clone()
                #         self.p_count += 1
                

                # Apply mask for updatable weights
                if self.updatable_weights < 1.0:
                    # if param_name == 'LHb.pre_w_pos': 
                    #     pre_p = p.data.clone()

                    # for each parameter, create a mask if it doesn't exist
                    if param_name not in self.mask_dict:
                        self.mask_dict[param_name] = (torch.rand_like(p.data) < self.updatable_weights).float()
                    # apply specific param mask to the update
                    p.data.addcdiv_(exp_avg * self.mask_dict[param_name], denom, value=-step_size)

                    # if param_name == 'LHb.pre_w_pos':
                    #     post_p = p.data.clone()
                    #     print(f"percent of unchanged values of original p.data and post update p.data: {(self.og_p == post_p).sum().item()/post_p.numel()}")
                    #     print(f"percent of unchanged values of pre update p.data and post update p.data: {(pre_p == post_p).sum().item()/post_p.numel()}")
                    #     print("first p.data recording:", self.og_p)
                    #     print("pre update p.data:", pre_p)
                    #     print("post update p.data:", post_p)
                else:
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

class TrainerMLE(object):
    """
    Maximum Likelihood Estimation (MLE) trainer for training probabilistic 
    models. 

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer used to update the model parameters during training.

    suppress_prints : bool, optional
        Whether to suppress print statements during training. Default is False.

    print_every : int, optional
        The frequency with which to print training updates. Default is 10.

    Methods
    -------

    train(model, train_data, valid_data, grad_clip_value=5, epochs=100,
            batch_size=256, shuffle=True, plot_losses=False, save_path=None)
            Trains the model on the training data and validates on the 
            validation data.

    Notes
    -----
      - The model must implement a `log_prob` method that computes the log 
        probability of observing the labels given the inputs.
      - train_data: Data object containing training data.
      - valid_data: Data object containing validation data.

    """
    
    def __init__(self, optimizer, suppress_prints=False, print_every=1):
        self.optimizer = optimizer
        self.suppress = suppress_prints
        self.print_every = print_every
    

    def train(
            self, 
            model, 
            train_data, 
            valid_data, 
            criterion,
            opto_category,
            grad_clip_value=5,
            epochs=5,
            batch_size=256, 
            shuffle=True, 
            load_best_state_dict=True,
            plot_losses=False, 
            save_path=None,
            ):
        """
        Trains a model using maximum likelihood estimation.

        Parameters
        ----------

        model : nn.Module
            The model to be trained.

        train_data : Dataset
            The training data.

        valid_data : Dataset
            The validation data.

        criterion : torch.nn.Module
            Loss function (e.g., nn.CrossEntropyLoss, nn.MSELoss).

        opto_category : int
            Category ID for optogenetic stimulation.

        grad_clip_value : float, optional
            The value at which to clip the gradients to prevent exploding 
            gradients. Default is 5.

        epochs : int, optional
            The number of epochs to train the model. Default is 5.

        batch_size : int, optional
            The batch size used during training. Default is 256.

        shuffle : bool, optional    
            Whether to shuffle the data during training. Default is True.

        load_best_state_dict : bool, optional
            Whether to load the best state dict at the end of training.
            Default is True.

        plot_losses : bool, optional
            Whether to plot the training and validation losses. 
            Default is False.

        save_path : str, optional
            Directory to save checkpoints and loss curves if provided.

        Returns
        -------

        dict
            A dictionary containing the best epoch, training losses, and
            validation losses. If `plot_losses` is True, the dictionary will
            also contain a figure object for the loss curves and the figure
            will be saved to `save_path` if provided.

        """

        # whether to save at checkpoints
        save_boolean = save_path is not None and os.path.exists(save_path)
        
        # send model to device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        # create dataloader objects for training data and validation data
        train_dataloader = DataLoader(train_data, 
                                      batch_size=batch_size, 
                                      shuffle=shuffle)
        valid_dataloader = DataLoader(valid_data, 
                                      batch_size=batch_size, 
                                      shuffle=shuffle)
        
        train_losses = []
        valid_losses = []
        train_accuracies = []
        valid_accuracies = []
        
        
        stime = time.time()
        
        best_loss = float('inf')
        best_state_dict = deepcopy(model.state_dict())

        for epoch in range(1, epochs + 1):        
            train_loss = 0
            valid_loss = 0
            distance_dict = {key: [0, 0] for key in range(10)}
            model.EP_current_act = None
            model.opto_idx_batch = None
            
            
            model.train()
            # Initialize neuron counter for tracking across batches
            neuron_counter_top = Counter()
            neuron_counter_bottom = Counter()
            
            # iterate over the training data
            for input_batch, (stim_label_batch, og_label_batch) in train_dataloader:
                model.opto_idx_batch = [z for z, x in enumerate(og_label_batch.tolist()) if x == opto_category]

                try:
                    input_batch = input_batch.to(device)
                    stim_label_batch = stim_label_batch.to(device)
                    og_label_batch = og_label_batch.to(device)
                    
                    outputs = model(input_batch)

                    # keep calculating top (opto) and lowest EP activation neurons every batch
                    model.EP_current_train_act = model.EP_current_act
                    model.opto_choose_neuron = choose_neuron(
                        model.EP_current_train_act, 
                        model.opto_idx_batch, 
                        True, 
                        neuron_counter=neuron_counter_top
                    )
                    model.lowest_choose_neuron = choose_neuron(
                        model.EP_current_train_act, 
                        model.opto_idx_batch, 
                        False,
                        neuron_counter=neuron_counter_bottom
                    )

                    loss = criterion(outputs, stim_label_batch)
                    
                    # zero gradients
                    self.optimizer.zero_grad()
                    # backpropagate loss
                    loss.backward()
                    
                    # prevent exploding gradients
                    clip_grad_value_(model.parameters(), grad_clip_value)
                    
                    # update weights
                    self.optimizer.step()

                    # Update signs dynamically after gradient computation
                    for name, module in model.named_modules():
                        if isinstance(module, TopKLinear):
                            module.update_signs()  # Flip signs if needed
                    
                    model.LHb.update_signs()
                    
                    # aggregate training loss
                    train_loss += loss.item()

                    
                
                except Exception as e:
                    print(f"Exception: {e}")
                    print("EXCEPTION IS HERE")
                    train_loss += 10

            model.EP_activations_ls = []
            model.opto_idx_trainset = []
            # compute mean training loss and save to list
            train_loss /= len(train_dataloader)
            train_losses.append(train_loss)
            
            model.eval()
            with torch.no_grad():
                # iterate over validation data
                for input_batch, (stim_label_batch, og_label_batch) in valid_dataloader:
                    try:
                        # produce negative log likelihood
                        outputs = model(input_batch.to(device))
                        loss = criterion(outputs, stim_label_batch)
                        
                        # compute and aggregate validation loss 
                        valid_loss += loss.item()
                        
                        # validation distance for each category
                        distance_dict = calculate_distance(stim_label_batch, og_label_batch, outputs, distance_dict)
                        
                    except Exception as e:
                        print(f"Exception: {e}")
                        valid_loss += 10
            
            # average distance dict
            distance_dict = {key: value[0] / value[1] for key, value in distance_dict.items()}
            
            # compute mean validation loss and save to list
            valid_loss /= len(valid_dataloader)
            valid_losses.append(valid_loss)

            # save model that performs best on validation data
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                best_state_dict = deepcopy(model.state_dict())
                best_distance_dict = distance_dict

                if save_boolean:
                    torch.save(best_state_dict, os.path.join(save_path, 'state_dict.pt'))

                    with open(os.path.join(save_path, 'losses.json'), 'w') as f:
                        json.dump(
                            {'best epoch': best_epoch,
                             'train losses': train_losses, 
                             'valid losses': valid_losses,
                            'best distance dict': best_distance_dict,
                            }
                        )

            if not self.suppress:
                # printing
                if epoch % self.print_every == 0:
                    print(f' ----   Epoch {epoch}  ---- ')
                    print(f'training loss: {train_loss:.2f} | validation loss: {valid_loss:.2f}')
                    
                    time_elapsed = time.time() - stime
                    pred_time_remaining = (time_elapsed / epoch) * (epochs - epoch)
                    
                    print(f'time elapsed: {time_elapsed:.2f} s | predicted time remaining: {pred_time_remaining:.2f} s')
                    
            if wandb.run is not None:
                wandb.log({"train_loss": float(train_loss), 
                           "valid_loss": float(valid_loss),
                           "epoch": epoch})
        
        results = {
            'best epoch': best_epoch, 
            'train losses': train_losses, 
            'valid losses': valid_losses,
            'best distance dict': best_distance_dict,
        }

        if load_best_state_dict:
            model.load_state_dict(best_state_dict)
        else:
            results['best state dict'] = best_state_dict
        
        return results