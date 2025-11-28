import torch
from torch import nn


TOPK_INIT_METHODS = {
    'xavier_normal': nn.init.xavier_normal_,
    'xavier_uniform': nn.init.xavier_uniform_,
    'kaiming_normal': nn.init.kaiming_normal_,
    'kaiming_uniform': nn.init.kaiming_uniform_,
    'orthogonal': nn.init.orthogonal_,
    'normal': nn.init.normal_,
    'uniform': nn.init.uniform_,
    'eye': nn.init.eye_,
}


class TopKLinear(nn.Module):
    """
    A linear layer that retains only the top K strongest synaptic weights for
    in_features.

    This module implements a linear transformation with weights constrained to 
    be positive. For each output neuron, only the top K weights are kept 
    , and the rest are set to zero. This simulates a neuron receiving 
    inputs only from its strongest synaptic connections.

    Parameters
    ----------

    in_features : int
        The number of input features.
    
    out_features : int
        The number of output features.

    K : int
        The number of strongest synapses to keep per dendritic branch.

    param_space : str, optional
        The parameter space for the weights. Options are 'log' and 'sigmoid'.
        If `'log'`, the weights are parameterized as exponentials of `pre_w`.
        If `'sigmoid'`, the weights are parameterized as sigmoids of `pre_w`.
        Defaults to 'log'.
    
    Attributes
    ----------

    pre_w : torch.nn.Parameter
        The raw weights before applying the exponential or sigmoid 
        transformation. Initialized with small negative values to ensure 
        positive weights after transformation.

    K : int
        The percentage of strongest synapses to keep per dendritic branch.

    param_space : str
        The parameter space for the weights.

    Methods
    -------

    forward(x)
        Performs a forward pass through the layer.

    weight()
        Returns the transformed synaptic weights after applying the exponential 
        or sigmoid.
    
    weight_mask()
        Returns a mask tensor indicating the top K synaptic connections per 
        output neuron.

    pruned_weight()
        Returns the pruned synaptic weights after applying the mask.

    weighted_synapses(cell_weights, prune=False)
        Returns the weighted synapses for a given set of cell

    Notes
    -----
    - All weights are constrained to be positive.
    - The prunning is done dynamically during the forward pass.

    Examples
    --------

    >>> import torch
    >>> from torch import nn
    >>> topk_linear = TopKLinear(in_features=10, out_features=5, K=3)
    >>> x = torch.randn(4, 10)  # Batch of 4 samples
    >>> output = topk_linear(x)
    >>> print(output.shape)
    torch.Size([4, 5])
    """
    def __init__(
        self,
        in_features,
        out_features,
        K,
        layer = None,
        param_space='log',
        init_method='xavier_normal',
        sign_pattern='mixed',  # Default: mixed signs
        update_sign=False,  # Default: sign matrix is not trainable
        dales_law=False,  # Default: no Dale's law
        pos_neg_layer=False
    ):
        super(TopKLinear, self).__init__()
        self.pos_neg_layer = pos_neg_layer
        if self.pos_neg_layer:
            self.pre_w_pos = nn.Parameter(
                torch.empty((out_features, in_features)), requires_grad=True)
            self.pre_w_neg = nn.Parameter(
                torch.empty((out_features, in_features)), requires_grad=True)

        else:
            self.pre_w = nn.Parameter(
                torch.empty((out_features, in_features)), requires_grad=True)

        self.sign_matrix = nn.Parameter(
            torch.empty((out_features, in_features)), requires_grad=False)
        
        self.K = round(K/100 * in_features)
        self.param_space = param_space
        self.init_method = init_method
        self.update_sign = update_sign # update sign based on gradient
        self.layer = layer
        self.sign_pattern = sign_pattern
        self.dales_law = dales_law

        self.initialize_signs(sign_pattern)
        self.initialize_weights()
        self.final_weight = self.weight().detach()
        

    def initialize_signs(self, sign_pattern):
        """Initialize the sign matrix based on the desired pattern."""
        if sign_pattern == 'mixed':
            self.sign_matrix.data.uniform_(-1, 1)
            self.sign_matrix.data = self.sign_matrix.sign()  # Values: -1 or 1
        elif sign_pattern == 'negative':
            self.sign_matrix.data.fill_(-1)  # All weights negative
        elif sign_pattern == 'positive':
            self.sign_matrix.data.fill_(1)  # All weights positive
        else:
            raise ValueError("Invalid sign pattern. Choose 'mixed', 'negative' or 'positive'.")

    def initialize_weights(self):
        """Initialize the raw weights using the specified method."""
        if self.init_method in TOPK_INIT_METHODS:
            init_func = TOPK_INIT_METHODS[self.init_method]
            if self.pos_neg_layer:
                init_func(self.pre_w_pos)
                init_func(self.pre_w_neg)
            else:
                init_func(self.pre_w)

            #TODO: Accept hyperparameters for initialization
            # init_func(self.pre_w, **self.init_params) 
        else:
            raise ValueError(
                f"Invalid initialization method: {self.init_method}. "
                f"Choose from {list(TOPK_INIT_METHODS.keys())}")


    def forward(self, x):
        """Perform the forward pass."""
        final_weight = self.weight()
        # matrix multiply inputs and synaptic weights
        mm = torch.mm(x, final_weight.t())
        
        # apply non-linearity
        return torch.sigmoid(mm)


        
    def weight(self):
        """Return the final weights."""
        if self.pos_neg_layer:
            weight_pos = self.transform_weight(self.pre_w_pos)
            weight_neg = self.transform_weight(self.pre_w_neg)
            # final weight = (+weight pos) + (-weight neg)
            final_weight = weight_pos - weight_neg
            if self.dales_law:
                final_weight = abs(weight_pos - weight_neg) * self.sign_matrix
            self.final_weight = (final_weight).detach()
            return final_weight
        else:
            final_weight = self.transform_weight(self.pre_w)
            if self.dales_law:
                final_weight = abs(final_weight) * self.sign_matrix
            self.final_weight = final_weight.detach()
            return final_weight
    
    def transform_weight(self, weight):
        """Return the transformed weights before matrix multiplication."""
        if self.param_space == 'log':
            trans_weight = weight.exp() * self.sign_matrix
        elif self.param_space == 'sigmoid':
            trans_weight = torch.sigmoid(weight)
        elif self.param_space == 'tanh':
            trans_weight = torch.tanh(weight)
        elif self.param_space == 'relu':
            trans_weight = torch.relu(weight)
        elif self.param_space == 'raw':
            trans_weight = weight

        return trans_weight
    
    def update_signs(self):
        """
        Dynamically switch signs based on gradients
        """
        if self.update_sign:
            with torch.no_grad():
                # Access gradients of log_weights
                grads = self.pre_w.grad  

                 # Compute the condition: W * grad(W) < 0
                 # i.e. gradient opposes weight direction
                flip_mask = (self.weight() * grads) < 0

                # Flip the signs to match the gradient
                self.sign_matrix[flip_mask] *= -1
    
    def decay_weights(self, weight_decay = 0.1):
        with torch.no_grad():
            self.pre_w.data -= (weight_decay * self.weight_mask())
    
    def weight_mask(self):
        """Generate a mask for the top K strongest connections."""
        topK_indices = torch.topk(self.pre_w, self.K, dim=-1, largest=True, sorted=False)[1]
        # initialize and populate masking matrix
        mask = torch.zeros_like(
            self.pre_w, device=self.pre_w.device, dtype=self.pre_w.dtype)
        mask[torch.arange(self.pre_w.shape[0])[:, None], topK_indices] = 1
        return mask


class Corelease_Model(nn.Module):
    """
    Model follows co-release: regular MLP

    if real: make DAN weights pure inhib
    """
    def __init__(self, in_features=784, h1=512, h2=512, out_features=11, dropout_rate=0, real=False, 
                 combine_EI=False, dales_law=False, opto_neuron_percent=0.4, batch_size=256, 
                 opto_on=False, log_weights=True):
        super().__init__()
        # dale's law
        if dales_law:
            self.EP = TopKLinear(in_features=in_features, out_features=h1, K=100, layer = "EP", param_space = "raw", sign_pattern="mixed", dales_law=True, pos_neg_layer=False)
            self.bn1 = nn.BatchNorm1d(h1)
            self.LHb = TopKLinear(in_features=h1, out_features=h2, K=100, layer = "LHb", param_space = "raw", sign_pattern="mixed", dales_law=True, pos_neg_layer=False)
            self.bn2 = nn.BatchNorm1d(h2)
            if dropout_rate != 0: self.dropout = nn.Dropout(dropout_rate)
            self.DAN = TopKLinear(in_features=h2, out_features=out_features,  K=100, layer = "DAN", param_space = "raw", sign_pattern="negative", dales_law=True, pos_neg_layer=False)

        # corelease
        else:
            self.EP = TopKLinear(in_features=in_features, out_features=h1, K=100, layer = "EP", param_space = "raw", sign_pattern="mixed", dales_law=False, pos_neg_layer=True)
            self.bn1 = nn.BatchNorm1d(h1)
            self.LHb = TopKLinear(in_features=h1, out_features=h2, K=100, layer = "LHb", param_space = "raw", sign_pattern="mixed", dales_law=False, pos_neg_layer=True)
            self.bn2 = nn.BatchNorm1d(h2)
            if dropout_rate != 0: self.dropout = nn.Dropout(dropout_rate)
            self.DAN = TopKLinear(in_features=h2, out_features=out_features,  K=100, layer = "DAN", param_space = "raw", sign_pattern="negative", dales_law=True, pos_neg_layer=False)


        self.opto_on = opto_on
        self.opto_neuron_percent = opto_neuron_percent
        self.init_weights = self.record_params(calc_sign=False)
        self.train_losses = []
        self.valid_losses = []
        
        # store opto idx
        self.opto_idx_batch = None
        self.EP_current_act = None
        self.EP_current_train_act = None
        self.opto_tag = False
        self.opto_choose_neuron = None
        self.lowest_choose_neuron = None


    def forward(self, x):
        """
        Forward pass with optional label handling for training.
        At Opto Activation:
        x size = ([batch_size, layer_size])
        set certain % of EP hidden layer neurons' activation values = 95th percentile of batch's activation values.
        """
        x = x.view(x.size(0), -1)  # Flatten input

        # Pass through the first layer, apply batch normalization and activation
        x = self.EP(x)  # Apply EP (TopKLinear or nn.Linear)
        self.EP_current_act = x
        
        x = self.bn1(x)  # Apply BatchNorm1d

        # Pass through the second layer, apply batch normalization and activation
        x = self.LHb(x)  # Apply LHb (TopKLinear or nn.Linear)
        x = self.bn2(x)  # Apply BatchNorm1d

        # Apply dropout for regularization
        if hasattr(self, 'dropout'):  
            x = self.dropout(x)

        # Pass through the final layer
        x = self.DAN(x)

        # Return logits for inference
        return x

    def record_params(self, calc_sign: bool = True, print_sign: bool = False):
        """ Save the network weights. """
        recorded_params = {}

        # Loop through all modules
        for name, module in self.named_modules():
            if isinstance(module, TopKLinear):  # Check if it's a TopKLinear layer
                # Store pre_w
                if module.pos_neg_layer:
                    if module.pre_w_pos.requires_grad:
                        with torch.no_grad():
                            recorded_params[name + '.pre_w_pos'] = module.pre_w_pos.data.detach().cpu().clone()
                    else:
                        recorded_params[name + '.pre_w_pos'] = module.pre_w_pos.data.detach().cpu().clone()

                    if module.pre_w_neg.requires_grad:
                        with torch.no_grad():
                            recorded_params[name + '.pre_w_neg'] = module.pre_w_neg.data.detach().cpu().clone()
                    else:
                        recorded_params[name + '.pre_w_neg'] = module.pre_w_neg.data.detach().cpu().clone()
                else:
                    if module.pre_w.requires_grad:
                        with torch.no_grad():
                            recorded_params[name + '.pre_w'] = module.pre_w.data.detach().cpu().clone()
                    else:
                        recorded_params[name + '.pre_w'] = module.pre_w.data.detach().cpu().clone()

                # Store sign_matrix
                if module.sign_matrix.requires_grad:
                    with torch.no_grad():
                        recorded_params[name + '.sign_matrix'] = module.sign_matrix.data.detach().cpu().clone()
                else:
                    recorded_params[name + '.sign_matrix'] = module.sign_matrix.data.detach().cpu().clone()

                # Store final_weight
                if module.final_weight.requires_grad:
                    with torch.no_grad():
                        recorded_params[name + '.final_weight'] = module.final_weight.data.detach().cpu().clone()
                else:
                    recorded_params[name + '.final_weight'] = module.final_weight.data.detach().cpu().clone()
        # Calculate the percentage of positive, negative, and zero weights
        if calc_sign:
            for name, cur_data in recorded_params.items():
                frac_pos = 100 * (torch.sum(cur_data > 0) / cur_data.numel()).numpy()
                frac_zero = 100 * (torch.sum(cur_data == 0) / cur_data.numel()).numpy()
                frac_neg = 100 * (torch.sum(cur_data < 0) / cur_data.numel()).numpy()
                print(name + f': Positive: {frac_pos:.2f}%; Negative: {frac_neg:.2f}%; Zero: {frac_zero:.2f}%')

        return recorded_params
    