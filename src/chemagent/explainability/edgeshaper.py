"""
EdgeSHAPer: Shapley Value Approximation for Edge Importance in Graph Neural Networks
====================================================================================

Module implementing the EdgeSHAPer algorithm for computing edge-level importance scores
in Graph Neural Networks using Monte Carlo sampling of Shapley values.

Key Concepts:
    - Shapley values: Game-theoretic fairness concept quantifying each player's (edge's)
      contribution to the outcome (prediction).
    - Edge masking: Iteratively remove/include edges from the graph and measure impact
      on model predictions.
    - Monte Carlo sampling: Approximate Shapley values through random permutations of edges.

Author: Andrea Mastropietro © All rights reserved
"""

import torch
import torch.nn.functional as F

import numpy as np
from numpy.random import default_rng

import matplotlib.pyplot as plt
# from tqdm import tqdm
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Draw

# edgeshaper_viz_utils wraps the visualization helpers used by EdgeSHAPer.
# Keep imports routed through that module so this file can still load when
# optional visualization dependencies are unavailable.
try:
    from chemagent.explainability.edgeshaper_viz_utils.molmapping import mapvalues2mol
    from chemagent.explainability.edgeshaper_viz_utils.utils import transform2png
    _HAS_EDGESHAPER_VIZ_UTILS = True
except Exception:
    _HAS_EDGESHAPER_VIZ_UTILS = False

###EdgeSHAPer as a class###

class Edgeshaper:
    """Compute Shapley value approximations for edge importance in GNNs.
    
    This class implements the EdgeSHAPer algorithm, which computes edge-level feature 
    importance scores for Graph Neural Network predictions using Monte Carlo-based 
    Shapley value approximation. The algorithm works by:
    
    1. Iterating over each edge in the graph
    2. For each edge, sampling random permutations of all edges
    3. Computing marginal contributions by comparing predictions with/without the edge
    4. Averaging marginal contributions across Monte Carlo samples
    
    The result is a real-valued score for each edge where:
        - Positive values: edges that boost the prediction
        - Negative values: edges that suppress the prediction
        - Near-zero values: edges with minimal impact
    
    Attributes:
        model (torch.nn.Module): Pre-trained GNN model (set to eval mode)
        x (torch.Tensor): Node feature matrix [num_nodes, num_features]
        edge_index (torch.Tensor): Graph connectivity [2, num_edges]
        edge_weight (torch.Tensor, optional): Edge weights [num_edges]
        device (str): Compute device ('cpu' or 'cuda')
        phi_edges (list): Computed Shapley values for each edge (set after explain)
        target_class (int): Class index being explained (set after explain)
        explained (bool): Flag indicating explanation has been computed
        pertinent_positive_set (torch.Tensor): Minimal edge subset preserving prediction class
        minimal_top_k_set (torch.Tensor): Minimal edge subset changing prediction
        fidelity (float): Fidelity+ metric (from minimal_top_k_set)
        infidelity (float): Fidelity- metric (from pertinent_positive_set)
        trustworthiness (float): Harmonic mean of fidelity and (1 - infidelity)
        original_pred_prob (float): Model's predicted probability for target class
    """

    def __init__(self, model, x, edge_index, edge_weight = None, device = "cpu"):
        """Initialize EdgeSHAPer explainer.
        
        Parameters
        ----------
        model : torch.nn.Module
            Pre-trained GNN model to explain (automatically set to eval mode)
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, num_features]
        edge_index : torch.Tensor
            Graph edge list of shape [2, num_edges]. First row: source nodes,
            second row: target nodes.
        edge_weight : torch.Tensor, optional
            Edge weight vector of shape [num_edges]. Default: None (uniform weights)
        device : str, optional
            Compute device ('cpu' or 'cuda'). Default: 'cpu'
        """
        super(Edgeshaper, self).__init__()
        # torch.manual_seed(12345)
        self.model = model
        self.model.to(device)
        self.x = x.to(device)
        self.edge_index = edge_index.to(device)
        self.edge_weight = edge_weight.to(device) if edge_weight is not None else None
        self.device = device


        self.phi_edges = None
        
        self.target_class = None
        self.explained = False

        self.pertinent_positive_set = None
        self.minimal_top_k_set = None
        self.fidelity = None
        self.infidelity = None
        self.trustuworthiness = None

        self.original_pred_prob = None

    def explain(self, M = 100, target_class = 0, P = None, deviation = None, log_odds = False, seed = None, progress_bar = True):
        """Compute Shapley values for edge importance (sequential Monte Carlo variant).
        
        Uses sequential computation (one Monte Carlo sample at a time). For large graphs,
        consider using explain_batch() instead for faster vectorized computation.
        
        Parameters
        ----------
        M : int, optional
            Number of Monte Carlo samples for Shapley approximation. More samples = 
            more accurate but slower. Default: 100
        target_class : int, optional
            Class index to explain (for classification models). For regression, 
            set to None. Default: 0
        P : float, optional
            Probability of edge existence in random subgraphs. If None, defaults to 
            the graph density (num_edges / max_possible_edges). Default: None
        deviation : float, optional
            Early stopping threshold. If set and the deviation between the current 
            Shapley sum and the true model output is ≤ deviation, stops early. 
            Ignores after M steps are reached. Default: None (no early stopping)
        log_odds : bool, optional
            If True, use log odds (raw model output) instead of softmax probabilities 
            for value computation. Default: False
        seed : int, optional
            Random seed for reproducibility. Default: None (unseeded)
        progress_bar : bool, optional
            Display tqdm progress bar during computation. Default: True
        
        Returns
        -------
        list
            Shapley values (phi_edges) for each edge, in the same order as edge_index.
            
        Notes
        -----
        The algorithm works as follows for each edge j:
        1. Sample M random edge permutations
        2. For each permutation, create two graphs:
           - With edge j (E_j_plus)
           - Without edge j (E_j_minus)
        3. Compute model predictions on both graphs
        4. Calculate marginal contribution: prediction_with - prediction_without
        5. Average across M samples to get phi_edges[j]
        """

        if deviation is not None:
            return self.explain_with_deviation(M = M, target_class = target_class, P = P, deviation = deviation, log_odds = log_odds, seed = seed, device=self.device)

        if target_class is None:
            print("No target class specified. Regression model assumed.")

        E = self.edge_index
        rng = default_rng(seed = seed)
        self.model.eval()
        phi_edges = []

        num_nodes = self.x.shape[0]
        num_edges = E.shape[1]
        
        if P == None:
            max_num_edges = num_nodes*(num_nodes-1)
            graph_density = num_edges/max_num_edges
            P = graph_density

        for j in tqdm(range(num_edges), disable = not progress_bar):
            marginal_contrib = 0
            for i in range(M):
                E_z_mask = rng.binomial(1, P, num_edges)
                E_mask = torch.ones(num_edges)
                pi = torch.randperm(num_edges)

                E_j_plus_index = torch.ones(num_edges, dtype=torch.int)
                E_j_minus_index = torch.ones(num_edges, dtype=torch.int)
                selected_edge_index = np.where(pi == j)[0].item()
                for k in range(num_edges):
                    if k <= selected_edge_index:
                        E_j_plus_index[pi[k]] = E_mask[pi[k]]
                    else:
                        E_j_plus_index[pi[k]] = E_z_mask[pi[k]]

                for k in range(num_edges):
                    if k < selected_edge_index:
                        E_j_minus_index[pi[k]] = E_mask[pi[k]]
                    else:
                        E_j_minus_index[pi[k]] = E_z_mask[pi[k]]


                #we compute marginal contribs
                
                # with edge j
                retained_indices_plus = torch.LongTensor(torch.nonzero(E_j_plus_index).tolist()).to(self.device).squeeze()
                E_j_plus = torch.index_select(E, dim = 1, index = retained_indices_plus)
                edge_weight_j_plus = None

                if self.edge_weight is not None:
                    edge_weight_j_plus = torch.index_select(self.edge_weight, dim = 0, index = retained_indices_plus)
                
                batch = torch.zeros(self.x.shape[0], dtype=int, device=self.x.device)
                
                out = self.model(self.x, E_j_plus, batch=batch, edge_weight=edge_weight_j_plus)
                out_prob = None
                V_j_plus = None
                
                if target_class is not None:
                    if not log_odds:
                        out_prob = F.softmax(out, dim = 1)
                    else:
                        out_prob = out #out prob variable now containts log_odds

                    V_j_plus = out_prob[0][target_class].item()
                else:
                    
                    out_prob = out #out prob variable now containts the regression output
                
                    V_j_plus = out_prob[0][0].item()

                # without edge j
                retained_indices_minus = torch.LongTensor(torch.nonzero(E_j_minus_index).tolist()).to(self.device).squeeze()
                E_j_minus = torch.index_select(E, dim = 1, index = retained_indices_minus)
                edge_weight_j_minus = None

                if self.edge_weight is not None:
                    edge_weight_j_minus = torch.index_select(self.edge_weight, dim = 0, index = retained_indices_minus)

                batch = torch.zeros(self.x.shape[0], dtype=int, device=self.x.device)
                out = self.model(self.x, E_j_minus, batch=batch, edge_weight=edge_weight_j_minus)

                V_j_minus = None
                
                if target_class is not None:
                    if not log_odds:
                        out_prob = F.softmax(out, dim = 1)
                    else:
                        out_prob = out #out prob variable now containts log_odds

                    V_j_minus = out_prob[0][target_class].item()
                else:
                    out_prob = out
                
                    V_j_minus = out_prob[0][0].item()

                marginal_contrib += (V_j_plus - V_j_minus)

            phi_edges.append(marginal_contrib/M)

        self.target_class = target_class
        self.explained = True    
        self.phi_edges = phi_edges
        return phi_edges


    def explain_with_deviation(self, M = 100, target_class = 0, P = None, deviation = None, log_odds = False, seed = None, device = "cpu"):
        
        if target_class is None:
            print("No target class specified. Regression model assumed.")
            
        rng = default_rng(seed = seed)
        self.model.eval()
        batch = torch.zeros(self.x.shape[0], dtype=int, device=self.x.device)
        E = self.edge_index
        out = self.model(self.x, E, batch=batch, edge_weight=self.edge_weight)
        out_prob_real = F.softmax(out, dim = 1)[0][target_class].item() if target_class is not None else out[0][0].item()

        num_nodes = self.x.shape[0]
        num_edges = E.shape[1]

        phi_edges = []
        phi_edges_current = [0] * num_edges
        
        if P == None:
            max_num_edges = num_nodes*(num_nodes-1)
            graph_density = num_edges/max_num_edges
            P = graph_density

        for i in tqdm(range(M)):
        
            for j in range(num_edges):
                
                
                E_z_mask = rng.binomial(1, P, num_edges)
                E_mask = torch.ones(num_edges)
                pi = torch.randperm(num_edges)

                E_j_plus_index = torch.ones(num_edges, dtype=torch.int)
                E_j_minus_index = torch.ones(num_edges, dtype=torch.int)
                selected_edge_index = np.where(pi == j)[0].item()
                for k in range(num_edges):
                    if k <= selected_edge_index:
                        E_j_plus_index[pi[k]] = E_mask[pi[k]]
                    else:
                        E_j_plus_index[pi[k]] = E_z_mask[pi[k]]

                for k in range(num_edges):
                    if k < selected_edge_index:
                        E_j_minus_index[pi[k]] = E_mask[pi[k]]
                    else:
                        E_j_minus_index[pi[k]] = E_z_mask[pi[k]]


                #we compute marginal contribs
                
                # with edge j
                retained_indices_plus = torch.LongTensor(torch.nonzero(E_j_plus_index).tolist()).to(device).squeeze()
                E_j_plus = torch.index_select(E, dim = 1, index = retained_indices_plus)
                edge_weight_j_plus = None

                if self.edge_weight is not None:
                    edge_weight_j_plus = torch.index_select(self.edge_weight, dim = 0, index = retained_indices_plus)

                batch = torch.zeros(self.x.shape[0], dtype=int, device=self.x.device)
                
                out = self.model(self.x, E_j_plus, batch=batch, edge_weight=edge_weight_j_plus)
                out_prob = None

                V_j_plus = None
                
                if target_class is not None:
                    if not log_odds:
                        out_prob = F.softmax(out, dim = 1)
                    else:
                        out_prob = out #out prob variable now containts log_odds

                    V_j_plus = out_prob[0][target_class].item()
                else:
                    
                    out_prob = out
                
                    V_j_plus = out_prob[0][0].item()

                # without edge j
                retained_indices_minus = torch.LongTensor(torch.nonzero(E_j_minus_index).tolist()).to(device).squeeze()
                E_j_minus = torch.index_select(E, dim = 1, index = retained_indices_minus)
                edge_weight_j_minus = None

                if self.edge_weight is not None:
                    edge_weight_j_minus = torch.index_select(self.edge_weight, dim = 0, index = retained_indices_minus)

                batch = torch.zeros(self.x.shape[0], dtype=int, device=self.x.device)
                out = self.model(self.x, E_j_minus, batch=batch, edge_weight=edge_weight_j_minus)

                V_j_minus = None
                
                if target_class is not None:
                    if not log_odds:
                        out_prob = F.softmax(out, dim = 1)
                    else:
                        out_prob = out #out prob variable now containts log_odds

                    V_j_minus = out_prob[0][target_class].item()
                else:
                    out_prob = out
                
                    V_j_minus = out_prob[0][0].item()

                phi_edges_current[j] += (V_j_plus - V_j_minus)

            
            phi_edges = [elem / (i+1) for elem in phi_edges_current]
            # print(sum(phi_edges))
            if abs(out_prob_real - sum(phi_edges)) <= deviation:
                break

        self.phi_edges = phi_edges
        self.target_class = target_class
        self.explained = True    
        return phi_edges

    def explain_batch(self, M = 100, target_class = 0, P = None, deviation = None, log_odds = False, seed = None, description = None, batch_size = 100, progress_bar = True):
        """Compute Shapley values for edge importance (vectorized batch variant).
        
        Faster variant that processes multiple Monte Carlo samples in parallel using
        vectorized tensor operations. Recommended for graphs with many edges.
        
        Parameters
        ----------
        M : int, optional
            Number of Monte Carlo samples. Will be auto-adjusted to be divisible by 
            batch_size if needed. Default: 100
        target_class : int, optional
            Class index to explain (required for classification). Must not be None.
            Default: 0
        P : float, optional
            Probability of edge existence in random subgraphs. If None, defaults to 
            graph density. Default: None
        deviation : float, optional
            **Note:** Not supported in batched mode. If provided, raises `Exception`.
            Use `explain()` instead for early stopping capability. Default: None
        log_odds : bool, optional
            If True, use log odds instead of softmax probabilities. Default: False
        seed : int, optional
            Random seed for reproducibility. Default: None
        description : str, optional
            Unused parameter (legacy). Default: None
        batch_size : int, optional
            Number of Monte Carlo samples to process simultaneously. Higher values 
            are faster but use more GPU memory. Default: 100
        progress_bar : bool, optional
            Display tqdm progress bar. Default: True
        
        Returns
        -------
        list
            Shapley values (phi_edges) for each edge in same order as edge_index.
        
        Raises
        ------
        Exception
            If target_class is None (batched mode requires classification target)
        Exception
            If deviation is not None (batched mode does not support early stopping)
        
        Notes
        -----
        Algorithm optimizations:
        - Vectorizes edge permutation generation across batch_size samples
        - Batches model inference to process multiple graphs simultaneously
        - Reduces loop overhead compared to sequential `explain()` variant
        - Memory trade-off: larger batch_size = faster but higher GPU memory usage
        """

        if deviation is not None:
            raise Exception("explaination with deviation not implemented")

        if target_class is None:
            raise Exception("Target class not specified")

        if M%batch_size != 0:
            batch_size = int(np.ceil(M / np.ceil(M / batch_size)))
            print("Adjusted batch size to {} to be a divisor of M".format(batch_size))


        rng = default_rng(seed = seed)
        rng_torch = torch.Generator()

        rng_torch.manual_seed(seed)

        self.model.eval()
        phi_edges = []

        num_nodes = self.x.shape[0]
        num_edges = self.edge_index.shape[1]

        # E -> [2,num_edges*batch_size] -> E = [E1,E2,...,E_batch_size]
        E = self.edge_index.repeat(1,batch_size)

        if P == None:
            max_num_edges = num_nodes*(num_nodes-1)
            graph_density = num_edges/max_num_edges
            P = graph_density

        for j in tqdm(range(num_edges), disable = not progress_bar):
            marginal_contrib = 0
            for i in range(int(M/batch_size)):
                # E_z_mask -> [batch_size*num_edges] -> E_z_mask = [E_z_mask_1,...,E_z_mask_batch_size]
                    # I don't use this version in order to have the classic and PARALLELIZE version with the same result
                    # E_z_mask = torch.tensor(rng.binomial(1, P, num_edges*batch_size), dtype=torch.int32)
                E_z_mask = torch.tensor([rng.binomial(1, P, num_edges) for _ in range(batch_size)], dtype=torch.int32).flatten()

                # E_mask -> [batch_size*num_edges] -> E_mask = [E_mask_1,...,E_mask_batch_size]
                E_mask = torch.ones(batch_size * num_edges, dtype=torch.int32)


                # pi -> [batch_size*num_edges]
                pi = torch.cat([torch.randperm(num_edges, generator = rng_torch) for _ in range(batch_size)], dim=0)
                # pi = torch.cat([torch.randperm(num_edges) for _ in range(batch_size)], dim=0)


                # E_j_plus_index -> [batch_size*num_edges] -> E_j_plus_index = [E_j_plus_index_1,...,E_j_plus_index_batch_size]
                E_j_plus_index = torch.ones(batch_size*num_edges, dtype=torch.int)
                # E_j_minus_index -> [batch_size*num_edges] -> E_j_minus_index = [E_j_minus_index_1,...,E_j_minus_index_batch_size]
                E_j_minus_index = torch.ones(batch_size*num_edges, dtype=torch.int)

                # selected_edge_index -> (array([id_1,id_2,...,id_batch_size])) (la presenza delle parentesi è importante)
                selected_edge_index = np.where(pi == j)

                # selected_edge_index -> [1, batch_size]
                selected_edge_index = torch.tensor(selected_edge_index)
                # selected_edge_index -> [batch_size]
                selected_edge_index = selected_edge_index.squeeze()
                # selected_edge_index -> [num_edges*batch_size] -> [id_1,...,id_1,id_2,...id_2,...,id_batch_size,...,id_batch_size]
                selected_edge_index = selected_edge_index.repeat_interleave(num_edges)

                # k_values -> [num_edges*batch_size] -> [0,1,...,num_edges,...,num_edges*(batch_size)-1]
                k_values = torch.arange(num_edges*batch_size)

                # add_to_pi -> [num_edges*batch_size] -> [0,...,0,num_edges,...,num_edges,...,(batch_size-1)*num_edges,...,(batch_size-1)*num_edges]
                        # add_to_pi = torch.tensor([i*num_edges for i in range(batch_size) for _ in range(num_edges)])
                add_to_pi = torch.arange(start=0, end=batch_size*num_edges, step=num_edges).repeat_interleave(num_edges)

                # pi contains batch_size permutations the edges, but in each batch the edge index are different hence we add an offset
                pi_add = pi + add_to_pi

                # also the indices of the nodes change through the different batches (hence we add an offset)

                    # add_to_edge_index = torch.tensor([i*self.x.shape[0] for i in range(batch_size)]).repeat_interleave(num_edges)
                # add_to_edge_index -> [batch_size] -> [0,num_nodes,2*num_nodes,...,(batch_size-1)*num_nodes]
                add_to_edge_index = torch.arange(start=0, end=batch_size*num_nodes, step=num_nodes)
                # add_to_edge_index -> [batch_size*num_edges] -> [0,...,0,num_nodes,...,num_edges,...,(batch_size-1)*num_nodes,...,(batch_size-1)*num_nodes]
                add_to_edge_index = add_to_edge_index.repeat_interleave(num_edges)
                # add_to_edge_index -> [2,batch_size*num_edges] ->
                #       [[0,...,0,num_nodes,...,num_edges,...,(batch_size-1)*num_nodes,...,(batch_size-1)*num_nodes],
                #       [0,...,0,num_nodes,...,num_edges,...,(batch_size-1)*num_nodes,...,(batch_size-1)*num_nodes]]
                add_to_edge_index = add_to_edge_index.repeat(2,1).to(self.device)

                E_j_plus_index[pi_add] = torch.where(k_values <= selected_edge_index, E_mask[pi_add], E_z_mask[pi_add])
                E_j_minus_index[pi_add] = torch.where(k_values < selected_edge_index, E_mask[pi_add], E_z_mask[pi_add])

                # we compute marginal contributions

                retained_indices_plus = torch.LongTensor(torch.nonzero(E_j_plus_index).tolist()).to(self.device).squeeze()

                # remember that E consider the same indices for the nodes of batch 1 and batch 2 and so on
                # hence we add the offset for each batch
                E_j_plus = torch.index_select(E+add_to_edge_index, dim = 1, index = retained_indices_plus)

                edge_weight_j_plus = None
                # edge_type_j_plus = None

                if self.edge_weight is not None:
                    edge_weight_j_plus = torch.index_select(self.edge_weight.repeat(batch_size,1), dim = 0, index = retained_indices_plus)
                #TODO: think about implementing edge type for heterogeneous graphs
                # if self.edge_type is not None:
                #     edge_type_j_plus  = torch.index_select(self.edge_type.repeat(batch_size), dim = 0, index = retained_indices_plus)

                # batch -> [batch_size] -> [0,1,...,batch_size-1]
                batch = torch.arange(batch_size, dtype=int, device=self.x.device)
                # batch -> [batch_size*num_nodes] -> [0,...,0,...,batch_size-1,...,batch_size-1]
                batch = batch.repeat_interleave(num_nodes)

                # out = self.model(self.x.repeat(batch_size,1), E_j_plus, batch=batch, edge_weight=edge_weight_j_plus, edge_type=edge_type_j_plus)
                out = self.model(self.x.repeat(batch_size,1), E_j_plus, batch=batch, edge_weight=edge_weight_j_plus)


                out_prob = None
                V_j_plus = None

                if target_class is not None:
                    if not log_odds:
                        out_prob = F.softmax(out, dim = 1)
                    else:
                        out_prob = out #out prob variable now containts log_odds

                    V_j_plus = torch.sum(out_prob[:,target_class]).item()

                # without edge j
                retained_indices_minus = torch.LongTensor(torch.nonzero(E_j_minus_index).tolist()).to(self.device).squeeze()

                E_j_minus = torch.index_select(E+add_to_edge_index, dim = 1, index = retained_indices_minus)

                edge_weight_j_minus = None
                # edge_type_j_minus = None

                if self.edge_weight is not None:
                    edge_weight_j_minus = torch.index_select(self.edge_weight.repeat(batch_size,1), dim = 0, index = retained_indices_minus)
                # if self.edge_type is not None:
                #     edge_type_j_minus  = torch.index_select(self.edge_type.repeat(batch_size), dim = 0, index = retained_indices_minus)

                # out = self.model(self.x.repeat(batch_size,1), E_j_minus, batch=batch, edge_weight=edge_weight_j_minus, edge_type=edge_type_j_minus)
                out = self.model(self.x.repeat(batch_size,1), E_j_minus, batch=batch, edge_weight=edge_weight_j_minus)

                V_j_minus = None

                if target_class is not None:
                    if not log_odds:
                        out_prob = F.softmax(out, dim = 1)
                    else:
                        out_prob = out #out prob variable now containts log_odds

                    V_j_minus = torch.sum(out_prob[:,target_class]).item()

                marginal_contrib += (V_j_plus - V_j_minus)

                print_stuff = False

            phi_edges.append(marginal_contrib/M)

        self.target_class = target_class
        self.explained = True
        self.phi_edges = phi_edges
        return phi_edges
        
    def compute_original_predicted_probability(self):
        """Compute model's predicted probability on the full original graph.
        
        This baseline is used to compute fidelity metrics. Stores result in 
        self.original_pred_prob for use in fidelity computations.
        
        Returns
        -------
        float
            Model's predicted probability for self.target_class on the full graph.
        """
        self.model.eval()
        batch = torch.zeros(self.x.shape[0], dtype=int, device=self.x.device)
        out_log_odds = self.model(self.x, self.edge_index, batch=batch, edge_weight=self.edge_weight)
        out_prob = F.softmax(out_log_odds, dim = 1)
        original_pred_prob = out_prob[0][self.target_class].item()

        self.original_pred_prob = original_pred_prob

        return self.original_pred_prob

    #TODO: check if egde_weights work correctly in the following two methods, if not modify them accordingly
    def compute_pertinent_positive_set(self, verbose = False):
        """Identify minimal edge subset that PRESERVES the predicted class (Fidelity-).
        
        Greedily selects the top-k most important edges (by Shapley value) until 
        the model's prediction keeps the same class as the original graph. Measures
        how much these edges alone preserve the prediction (fidelity-).
        
        Parameters
        ----------
        verbose : bool, optional
            If True, print fidelity- metric. Default: False
        
        Returns
        -------
        tuple
            (edge_index, infidelity) where:
            - edge_index (torch.Tensor): Minimal edge subset with shape [2, k]
            - infidelity (float): Fidelity- = original_prob - reduced_prob.
              Lower values indicate the edges alone capture more of the original prediction.
        
        Notes
        -----
        Fidelity- (infidelity) measures how faithful the minimal set is to the original.
        - High fidelity- → important edges alone don't fully preserve the prediction
        - Low fidelity- → important edges alone capture most of the original prediction
        """
        assert(self.explained) #make sure that the explanation has been computed
        
        if self.target_class is None:
            raise Exception("Minimal informative sets are not defined for regression problems.")

        if self.original_pred_prob is None:
            self.compute_original_predicted_probability()

        self.model.eval()
        infidelity = 1 #None it was none, last edit since it remains none if the class does not change, so we put 1
        important_edges_ranking = np.argsort(-np.array(self.phi_edges))
        for i in range(1, important_edges_ranking.shape[0]+1):
            reduced_edge_index = torch.index_select(self.edge_index, dim = 1, index = torch.LongTensor(important_edges_ranking[0:i]).to(self.device))
            batch = torch.zeros(self.x.shape[0], dtype=int, device=self.x.device)
            out = self.model(self.x, reduced_edge_index, batch=batch)
            out_prob = F.softmax(out, dim = 1)
            # print(out_prob)
            predicted_class = torch.argmax(out_prob[0]).item()
            if (predicted_class == self.target_class):
                
                
                pred_prob = out_prob[0][self.target_class].item()
                infidelity = self.original_pred_prob-pred_prob

                if verbose:
                    print("FID- using pertinent positive set: ", infidelity)
                break

        self.pertinent_positive_set = reduced_edge_index
        self.infidelity = infidelity
        return reduced_edge_index, infidelity

    def compute_minimal_top_k_set(self, verbose = False):
        """Identify minimal edge subset that CHANGES the predicted class (Fidelity+).
        
        Greedily removes edges in order of decreasing Shapley value until the 
        model's prediction changes from the original class. Measures how much of 
        the prediction drop is due to removing edges (fidelity+).
        
        Parameters
        ----------
        verbose : bool, optional
            If True, print fidelity+ metric. Default: False
        
        Returns
        -------
        tuple
            (edge_index, fidelity) where:
            - edge_index (torch.Tensor): Edge subset with shape [2, k]
            - fidelity (float): Fidelity+ = original_prob - reduced_prob.
              Higher values indicate these edges are more important for the prediction.
        
        Notes
        -----
        Fidelity+ (sufficiency) measures how sufficient the edges are to drive the
        prediction toward the target class:
        - High fidelity+ → edges collectively have large impact on prediction
        - Low fidelity+ → edges have modest impact, other features also matter
        """
        assert(self.explained) #make sure that the explanation has been computed
        
        if self.target_class is None:
            raise Exception("Minimal informative sets are not defined for regression problems.")
            
        if self.original_pred_prob is None:
            self.compute_original_predicted_probability()

        self.model.eval()
        fidelity = 0 #it was None, last edit since it remains none if the class does not change, so we put 0
        pertinent_set_indices = []
        pertinent_set_edge_index = None
        important_edges_ranking = np.argsort(-np.array(self.phi_edges))
        for i in range(important_edges_ranking.shape[0]):
            index_of_edge_to_remove = important_edges_ranking[i]
            pertinent_set_indices.append(index_of_edge_to_remove)

            reduced_edge_index = torch.index_select(self.edge_index, dim = 1, index = torch.LongTensor(important_edges_ranking[i:]).to(self.device))
            
            # all nodes belong to same graph
            batch = torch.zeros(self.x.shape[0], dtype=int, device=self.x.device)
            out = self.model(self.x, reduced_edge_index, batch=batch)
            out_prob = F.softmax(out, dim = 1)
            # print(out_prob)
            predicted_class = torch.argmax(out_prob[0]).item()

            if predicted_class != self.target_class:
                pred_prob = out_prob[0][self.target_class].item()
                fidelity = self.original_pred_prob - pred_prob
                if verbose:
                    print("FID+ using minimal top-k set: ", fidelity)
                break

        pertinent_set_edge_index = torch.index_select(self.edge_index, dim = 1, index = torch.LongTensor(pertinent_set_indices).to(self.device))
        
        self.minimal_top_k_set = pertinent_set_edge_index
        self.fidelity = fidelity

        return pertinent_set_edge_index, fidelity


    def compute_trustworthiness(self, verbose = False):
        """Compute overall trustworthiness score of the explanation.
        
        Combines fidelity+ and fidelity- metrics into a single trustworthiness 
        score using harmonic mean. Captures both the importance of edges for the 
        prediction and their ability to preserve it.
        
        Parameters
        ----------
        verbose : bool, optional
            If True, print trustworthiness score. Default: False
        
        Returns
        -------
        float
            Trustworthiness score in range [0, 1]:
            TW = 2 * (fidelity * (1 - infidelity)) / (fidelity + (1 - infidelity))
            - TW = 1: Perfect explanation (high fidelity, low infidelity)
            - TW = 0: Poor explanation (either low fidelity or high infidelity)
        
        Notes
        -----
        Trustworthiness combines:
        - Fidelity+: How much important edges contribute to the prediction
        - Infidelity: How much do important edges preserve the original prediction
        
        Requires both fidelity and infidelity to be computed first.
        """

        assert(self.explained) #make sure that the explanation has been computed

        assert(self.fidelity is not None) #make sure that the fidelity has been computed
        assert(self.infidelity is not None) #make sure that the infidelity has been computed

        TW = None
        
        if self.fidelity+(1-self.infidelity) == 0:
            TW = 0
        else:
            TW = 2* ((self.fidelity*(1-self.infidelity))/(self.fidelity+(1-self.infidelity)))

        self.trustworthiness = TW

        if verbose:
            print("Trustworthiness: ", self.trustuworthiness)

        return self.trustworthiness

    def visualize_molecule_explanations(self, smiles, save_path=None, pertinent_positive = False, minimal_top_k = False):
        """Render Shapley values as heatmaps on molecular structure.
        
        Creates RDKit-based heatmap visualizations showing edge importance scores 
        mapped onto molecular bonds. Can visualize: (1) all edge Shapley values, 
        (2) only pertinent positive set edges, or (3) only minimal top-k set edges.
        
        Parameters
        ----------
        smiles : str
            SMILES string of the molecule to render
        save_path : str, optional
            Directory to save PNG files. If None, images not saved. Default: None
        pertinent_positive : bool, optional
            If True, visualize pertinent_positive_set (edges that preserve prediction). 
            Default: False
        minimal_top_k : bool, optional
            If True, visualize minimal_top_k_set (edges that change prediction). 
            Default: False
        
        Returns
        -------
        tuple
            (img_expl, img_pert_pos, img_min_top_k) where each is PIL.Image or None:
            - img_expl: Heatmap of all Shapley values
            - img_pert_pos: Heatmap of pertinent_positive_set (if pertinent_positive=True)
            - img_min_top_k: Heatmap of minimal_top_k_set (if minimal_top_k=True)
        
        Notes
        -----
        Maps graph edge indices to RDKit bond indices using atom connectivity.
        Aggregates multiple graph edges to single molecular bonds when present.
        """
        assert(self.explained) #make sure that the explanation has been computed

        if not _HAS_EDGESHAPER_VIZ_UTILS:
            raise ImportError(
                "EdgeSHAPer visualization helpers are unavailable. Check the edgeshaper_viz_utils dependencies to enable visualization."
            )

        img_expl = None
        img_pert_pos = None
        img_min_top_k = None

        edge_index = self.edge_index.to("cpu")

        test_mol = Chem.MolFromSmiles(smiles)
        test_mol = Draw.PrepareMolForDrawing(test_mol)

        num_bonds = len(test_mol.GetBonds())

        rdkit_bonds = {}

        for i in range(num_bonds):
            init_atom = test_mol.GetBondWithIdx(i).GetBeginAtomIdx()
            end_atom = test_mol.GetBondWithIdx(i).GetEndAtomIdx()
            
            rdkit_bonds[(init_atom, end_atom)] = i

        rdkit_bonds_phi = [0]*num_bonds
        for i in range(len(self.phi_edges)):
            phi_value = self.phi_edges[i]
            init_atom = edge_index[0][i].item()
            end_atom = edge_index[1][i].item()
            
            if (init_atom, end_atom) in rdkit_bonds:
                bond_index = rdkit_bonds[(init_atom, end_atom)]
                rdkit_bonds_phi[bond_index] += phi_value
            if (end_atom, init_atom) in rdkit_bonds:
                bond_index = rdkit_bonds[(end_atom, init_atom)]
                rdkit_bonds_phi[bond_index] += phi_value

        plt.clf()
        canvas = mapvalues2mol(test_mol, None, rdkit_bonds_phi, atom_width=0.2, bond_length=0.5, bond_width=0.5) #TBD: only one direction for edges? bonds weights is wrt rdkit bonds order?
        img_expl = transform2png(canvas.GetDrawingText())

        if save_path is not None:
            img_expl.save(save_path + "/" + "EdgeSHAPer_explanations_heatmap.png", dpi = (300,300))

        if pertinent_positive:
            if self.pertinent_positive_set is None:
                self.compute_pertinent_positivite_set()

            rdkit_bonds_phi_pertinent_set = [0]*num_bonds
            pertinent_set_edge_index = self.pertinent_positive_set
            for i in range(pertinent_set_edge_index.shape[1]):
                
                init_atom = pertinent_set_edge_index[0][i].item()
                end_atom = pertinent_set_edge_index[1][i].item()
                
                
                if (init_atom, end_atom) in rdkit_bonds:
                    bond_index = rdkit_bonds[(init_atom, end_atom)]
                    if rdkit_bonds_phi_pertinent_set[bond_index] == 0:
                        rdkit_bonds_phi_pertinent_set[bond_index] += rdkit_bonds_phi[bond_index]
                if (end_atom, init_atom) in rdkit_bonds:
                    bond_index = rdkit_bonds[(end_atom, init_atom)]
                    if rdkit_bonds_phi_pertinent_set[bond_index] == 0:
                        rdkit_bonds_phi_pertinent_set[bond_index] += rdkit_bonds_phi[bond_index]

            plt.clf()
            canvas = mapvalues2mol(test_mol, None, rdkit_bonds_phi_pertinent_set, atom_width=0.2, bond_length=0.5, bond_width=0.5)
            img_pert_pos = transform2png(canvas.GetDrawingText())

            if save_path is not None:

                img_pert_pos.save(save_path + "/" + "EdgeSHAPer_pertinent_positive_set_heatmap.png", dpi = (300,300))

        if minimal_top_k:
            if self.minimal_top_k_set is None:
                self.compute_minimal_top_k_set()

            rdkit_bonds_phi_pertinent_set = [0]*num_bonds
            pertinent_set_edge_index = self.minimal_top_k_set
            for i in range(pertinent_set_edge_index.shape[1]):
                
                init_atom = pertinent_set_edge_index[0][i].item()
                end_atom = pertinent_set_edge_index[1][i].item()
                
                
                if (init_atom, end_atom) in rdkit_bonds:
                    bond_index = rdkit_bonds[(init_atom, end_atom)]
                    if rdkit_bonds_phi_pertinent_set[bond_index] == 0:
                        rdkit_bonds_phi_pertinent_set[bond_index] += rdkit_bonds_phi[bond_index]
                if (end_atom, init_atom) in rdkit_bonds:
                    bond_index = rdkit_bonds[(end_atom, init_atom)]
                    if rdkit_bonds_phi_pertinent_set[bond_index] == 0:
                        rdkit_bonds_phi_pertinent_set[bond_index] += rdkit_bonds_phi[bond_index]

            plt.clf()
            canvas = mapvalues2mol(test_mol, None, rdkit_bonds_phi_pertinent_set, atom_width=0.2, bond_length=0.5, bond_width=0.5)
            img_min_top_k = transform2png(canvas.GetDrawingText())

            if save_path is not None:
                img_min_top_k.save(save_path + "/" + "EdgeSHAPer_minimal_top_k_set_heatmap.png", dpi = (300,300))

        plt.clf()
        return img_expl, img_pert_pos, img_min_top_k    


    


###EdgeSHAPer as a function###

def edgeshaper(model, x, E, M = 100, target_class = 0, P = None, deviation = None, log_odds = False, seed = 42, edge_weight = None, device = "cpu"):
    """Functional interface for EdgeSHAPer (sequential Monte Carlo variant).
    
    Standalone function for computing edge Shapley values. For more features 
    (e.g., fidelity metrics, visualizations), use the Edgeshaper class instead.
    
    Parameters
    ----------
    model : torch.nn.Module
        Pre-trained GNN model in eval mode
    x : torch.Tensor
        Node feature matrix [num_nodes, num_features]
    E : torch.Tensor
        Edge index [2, num_edges]
    M : int, optional
        Monte Carlo samples for Shapley approximation. Default: 100
    target_class : int, optional
        Class index to explain (classification). Default: 0
    P : float, optional
        Edge existence probability in random subgraphs. If None, uses graph density.
        Default: None
    deviation : float, optional
        Early stopping threshold. If set and deviation ≤ threshold, stops early.
        Uses explain_with_deviation() internally. Default: None
    log_odds : bool, optional
        Use log odds instead of softmax probabilities. Default: False
    seed : int, optional
        Random seed for reproducibility. Default: 42
    edge_weight : torch.Tensor, optional
        Edge weights [num_edges]. Default: None (uniform)
    device : str, optional
        Compute device ('cpu' or 'cuda'). Default: 'cpu'
    
    Returns
    -------
    list
        Shapley values for each edge, in same order as E (edge_index)
    
    See Also
    --------
    Edgeshaper : Class-based interface with more features (fidelity, visualization)
    edgeshaper_deviation : Implementation with early stopping support
    """
    
    if deviation != None:
        return edgeshaper_deviation(model, x, E, M = M, target_class = target_class, P = P, deviation = deviation, log_odds = log_odds, seed = seed, edge_weight = edge_weight, device=device)


    rng = default_rng(seed = seed)
    model.eval()
    phi_edges = []

    num_nodes = x.shape[0]
    num_edges = E.shape[1]
    
    if P == None:
        max_num_edges = num_nodes*(num_nodes-1)
        graph_density = num_edges/max_num_edges
        P = graph_density

    for j in tqdm(range(num_edges)):
        marginal_contrib = 0
        for i in range(M):
            E_z_mask = rng.binomial(1, P, num_edges)
            E_mask = torch.ones(num_edges)
            pi = torch.randperm(num_edges)

            E_j_plus_index = torch.ones(num_edges, dtype=torch.int)
            E_j_minus_index = torch.ones(num_edges, dtype=torch.int)
            selected_edge_index = np.where(pi == j)[0].item()
            for k in range(num_edges):
                if k <= selected_edge_index:
                    E_j_plus_index[pi[k]] = E_mask[pi[k]]
                else:
                    E_j_plus_index[pi[k]] = E_z_mask[pi[k]]

            for k in range(num_edges):
                if k < selected_edge_index:
                    E_j_minus_index[pi[k]] = E_mask[pi[k]]
                else:
                    E_j_minus_index[pi[k]] = E_z_mask[pi[k]]


            #we compute marginal contribs
            
            # with edge j
            retained_indices_plus = torch.LongTensor(torch.nonzero(E_j_plus_index).tolist()).to(device).squeeze()
            E_j_plus = torch.index_select(E, dim = 1, index = retained_indices_plus)
            edge_weight_j_plus = None

            if edge_weight is not None:
                edge_weight_j_plus = torch.index_select(edge_weight, dim = 0, index = retained_indices_plus)

            batch = torch.zeros(x.shape[0], dtype=int, device=x.device)
            
            out = model(x, E_j_plus, batch=batch, edge_weight = edge_weight_j_plus)
            out_prob = None

            V_j_plus = None
                
            if target_class is not None:
                if not log_odds:
                    out_prob = F.softmax(out, dim = 1)
                else:
                    out_prob = out #out prob variable now containts log_odds

                V_j_plus = out_prob[0][target_class].item()
            else:
                
                out_prob = out
            
                V_j_plus = out_prob[0][0].item()

            # without edge j
            retained_indices_minus = torch.LongTensor(torch.nonzero(E_j_minus_index).tolist()).to(device).squeeze()
            E_j_minus = torch.index_select(E, dim = 1, index = retained_indices_minus)
            edge_weight_j_minus = None

            if edge_weight is not None:
                edge_weight_j_minus = torch.index_select(edge_weight, dim = 0, index = retained_indices_minus)

            batch = torch.zeros(x.shape[0], dtype=int, device=x.device)
            out = model(x, E_j_minus, batch=batch, edge_weight = edge_weight_j_minus)

            V_j_minus = None
                
            if target_class is not None:
                if not log_odds:
                    out_prob = F.softmax(out, dim = 1)
                else:
                    out_prob = out #out prob variable now containts log_odds

                V_j_minus = out_prob[0][target_class].item()
            else:
                out_prob = out
            
                V_j_minus = out_prob[0][0].item()

            marginal_contrib += (V_j_plus - V_j_minus)

        phi_edges.append(marginal_contrib/M)
        
    return phi_edges


def edgeshaper_deviation(model, x, E, M = 100, target_class = 0, P = None, deviation = None, log_odds = False, seed = 42, edge_weight = None, device = "cpu"):
    """Compute edge Shapley values with early stopping (deviation-based variant).
    
    Internal implementation supporting early stopping. Stops sampling once the 
    cumulative Shapley value sum deviates from the true model output by at most 
    'deviation'. Otherwise runs full M iterations.
    
    Parameters
    ----------
    model : torch.nn.Module
        Pre-trained GNN model in eval mode
    x : torch.Tensor
        Node feature matrix [num_nodes, num_features]
    E : torch.Tensor
        Edge index [2, num_edges]
    M : int, optional
        Maximum number of Monte Carlo samples. Default: 100
    target_class : int, optional
        Class index for classification. Default: 0
    P : float, optional
        Edge existence probability. If None, uses graph density. Default: None
    deviation : float, optional
        Early stopping threshold: stops when |sum(phi_edges) - model_output| <= deviation.
        Default: None (runs full M iterations)
    log_odds : bool, optional
        Use log odds instead of softmax. Default: False
    seed : int, optional
        Random seed. Default: 42
    edge_weight : torch.Tensor, optional
        Edge weights [num_edges]. Default: None
    device : str, optional
        Compute device ('cpu' or 'cuda'). Default: 'cpu'
    
    Returns
    -------
    list
        Shapley values for each edge
    
    Notes
    -----
    Early stopping can significantly speed up computation but may sacrifice accuracy.
    Set deviation to a reasonable threshold (e.g., 0.01 for 1% error tolerance).
    
    See Also
    --------
    edgeshaper : Functional interface (calls this if deviation is set)
    Edgeshaper.explain_with_deviation : Class method variant
    """
    model.eval()
    batch = torch.zeros(x.shape[0], dtype=int, device=x.device)
    out = model(x, E, batch=batch, edge_weight = edge_weight)
    out_prob_real = F.softmax(out, dim = 1)[0][target_class].item() if target_class is not None else out[0][0].item()

    num_nodes = x.shape[0]
    num_edges = E.shape[1]

    phi_edges = []
    phi_edges_current = [0] * num_edges
    
    if P == None:
        max_num_edges = num_nodes*(num_nodes-1)
        graph_density = num_edges/max_num_edges
        P = graph_density

    for i in tqdm(range(M)):
    
        for j in range(num_edges):
            
            
            E_z_mask = rng.binomial(1, P, num_edges)
            E_mask = torch.ones(num_edges)
            pi = torch.randperm(num_edges)

            E_j_plus_index = torch.ones(num_edges, dtype=torch.int)
            E_j_minus_index = torch.ones(num_edges, dtype=torch.int)
            selected_edge_index = np.where(pi == j)[0].item()
            for k in range(num_edges):
                if k <= selected_edge_index:
                    E_j_plus_index[pi[k]] = E_mask[pi[k]]
                else:
                    E_j_plus_index[pi[k]] = E_z_mask[pi[k]]

            for k in range(num_edges):
                if k < selected_edge_index:
                    E_j_minus_index[pi[k]] = E_mask[pi[k]]
                else:
                    E_j_minus_index[pi[k]] = E_z_mask[pi[k]]


            #we compute marginal contribs
            
            # with edge j
            retained_indices_plus = torch.LongTensor(torch.nonzero(E_j_plus_index).tolist()).to(device).squeeze()
            E_j_plus = torch.index_select(E, dim = 1, index = retained_indices_plus)

            edge_weight_j_plus = None

            if edge_weight is not None:
                edge_weight_j_plus = torch.index_select(edge_weight, dim = 0, index = retained_indices_plus)

            batch = torch.zeros(x.shape[0], dtype=int, device=x.device)
            
            out = model(x, E_j_plus, batch=batch, edge_weight = edge_weight_j_plus)
            out_prob = None

            V_j_plus = None
                
            if target_class is not None:
                if not log_odds:
                    out_prob = F.softmax(out, dim = 1)
                else:
                    out_prob = out #out prob variable now containts log_odds

                V_j_plus = out_prob[0][target_class].item()
            else:
                out_prob = out
            
                V_j_plus = out_prob[0][0].item()

            # without edge j
            retained_indices_minus = torch.LongTensor(torch.nonzero(E_j_minus_index).tolist()).to(device).squeeze()
            E_j_minus = torch.index_select(E, dim = 1, index = retained_indices_minus)

            edge_weight_j_minus = None

            if edge_weight is not None:
                edge_weight_j_minus = torch.index_select(edge_weight, dim = 0, index = retained_indices_minus)

            batch = torch.zeros(x.shape[0], dtype=int, device=x.device)
            out = model(x, E_j_minus, batch=batch, edge_weight = edge_weight_j_minus)

            V_j_minus = None
                
            if target_class is not None:
                if not log_odds:
                    out_prob = F.softmax(out, dim = 1)
                else:
                    out_prob = out #out prob variable now containts log_odds

                V_j_minus = out_prob[0][target_class].item()
            else:
                out_prob = out
            
                V_j_minus = out_prob[0][0].item()

            phi_edges_current[j] += (V_j_plus - V_j_minus)

        
        phi_edges = [elem / (i+1) for elem in phi_edges_current]
        # print(sum(phi_edges))
        if abs(out_prob_real - sum(phi_edges)) <= deviation:
            break
             
    return phi_edges


def create_edge_index(mol, weighted=False):
    """
    Create edge index for a molecule.
    """
    adj = nx.to_scipy_sparse_array(mol).tocoo() #nx.to_scipy_sparse_matrix(mol).tocoo()
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)

    if weighted:
        weights = torch.from_numpy(adj.data.astype(np.float32))
        edge_weight = torch.FloatTensor(weights)
        return edge_index, edge_weight

    return edge_index
