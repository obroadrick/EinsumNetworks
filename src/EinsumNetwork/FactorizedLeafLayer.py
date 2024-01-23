from EinsumNetwork.Layer import Layer
from EinsumNetwork.ExponentialFamilyArray import *


class FactorizedLeafLayer(Layer):
    """
    Computes all EiNet leaves in parallel, where each leaf is a vector of factorized distributions, where factors are
    from exponential families.

    In FactorizedLeafLayer, we generate an ExponentialFamilyArray with array_shape = (num_dist, num_replica), where
        num_dist is the vector length of the vectorized distributions (K in the paper), and
        num_replica is picked large enough such that "we compute enough leaf densities". At the moment we rely that
            the PC structure (see Class Graph) provides the necessary information to determine num_replica. In
            particular, we require that each leaf of the graph has the field einet_address.replica_idx defined;
            num_replica is simply the max over all einet_address.replica_idx.
            In the future, it would convenient to have an automatic allocation of leaves to replica, without requiring
            the user to specify this.
    The generate ExponentialFamilyArray has shape (batch_size, num_var, num_dist, num_replica). This array of densities
    will contain all densities over single RVs, which are then multiplied (actually summed, due to log-domain
    computation) together in forward(...).
    """

    def __init__(self, leaves, num_var, num_dims, exponential_family, ef_args, use_em=True, roth=False):
        """
        :param leaves: list of PC leaves (DistributionVector, see Graph.py)
        :param num_var: number of random variables (int)
        :param num_dims: dimensionality of RVs (int)
        :param exponential_family: type of exponential family (derived from ExponentialFamilyArray)
        :param ef_args: arguments of exponential_family
        :param use_em: use on-board EM algorithm? (boolean)
        """
        super(FactorizedLeafLayer, self).__init__(use_em=use_em)

        self.nodes = leaves
        self.num_var = num_var
        self.num_dims = num_dims

        self.roth = roth

        num_dist = list(set([n.num_dist for n in self.nodes]))
        if len(num_dist) != 1:
            raise AssertionError("All leaves must have the same number of distributions.")
        num_dist = num_dist[0]
        
        self.num_dist = num_dist# oliver added this ... this is K in the paper

        replica_indices = set([n.einet_address.replica_idx for n in self.nodes])
        if sorted(list(replica_indices)) != list(range(len(replica_indices))):
            raise AssertionError("Replica indices should be consecutive, starting with 0.")
        num_replica = len(replica_indices)

        if self.roth:
            self.coefs = torch.nn.Parameter(torch.randn((len(self.nodes), 1+self.num_var, self.num_dist)), requires_grad=True)
            self.coefs_bar = torch.nn.Parameter(torch.randn((len(self.nodes), 1+self.num_var, self.num_dist)), requires_grad=True)
            # create mask for scopes (to be used in forward)
            self.scopes_mask = []
            for c, node in enumerate(self.nodes):
                self.scopes_mask.append(torch.cat((torch.LongTensor([1.0]),torch.zeros(self.num_var,dtype=torch.long).scatter(0,torch.LongTensor(list(node.scope)), 1))))
            self.scopes_mask = torch.stack(self.scopes_mask)
            self.scope_sizes = torch.sum(self.scopes_mask,1)
        else:
            # this computes an array of (batch, num_var, num_dist, num_repetition) exponential family densities
            # see ExponentialFamilyArray
            self.ef_array = exponential_family(num_var, num_dims, (num_dist, num_replica), use_em=use_em, **ef_args)

        # self.scope_tensor indicates which densities in self.ef_array belongs to which leaf.
        # TODO: it might be smart to have a sparse implementation -- I have experimented a bit with this, but it is not
        # always faster.
        self.register_buffer('scope_tensor', torch.zeros((num_var, num_replica, len(self.nodes))))
        for c, node in enumerate(self.nodes):
            self.scope_tensor[node.scope, node.einet_address.replica_idx, c] = 1.0
            node.einet_address.layer = self
            node.einet_address.idx = c

    # --------------------------------------------------------------------------------
    # Implementation of Layer interface

    def default_initializer(self):
        if not self.roth:
            return self.ef_array.default_initializer()

    def initialize(self, initializer=None):
        if self.roth:
            return
            #todo
        else:
            self.ef_array.initialize(initializer)

    def forward(self, x=None):
        """
        Compute the factorized leaf densities. We are doing the computation in the log-domain, so this is actually
        computing sums over densities.

        We first pass the data x into self.ef_array, which computes a tensor of shape
        (batch_size, num_var, num_dist, num_replica). This is best interpreted as vectors of length num_dist, for each
        sample in the batch and each RV. Since some leaves have overlapping scope, we need to compute "enough" leaves,
        hence the num_replica dimension. The assignment of these log-densities to leaves is represented with
        self.scope_tensor.
        In the end, the factorization (sum in log-domain) is realized with a single einsum.

        :param x: input data (Tensor).
                  If self.num_dims == 1, this can be either of shape (batch_size, self.num_var, 1) or
                  (batch_size, self.num_var).
                  If self.num_dims > 1, this must be of shape (batch_size, self.num_var, self.num_dims).
        :return: log-density vectors of leaves
                 Will be of shape (batch_size, num_dist, len(self.nodes))   ...   (batch_size, K, num_leaves)
                 Note: num_dist is K in the paper, len(self.nodes) is the number of PC leaves
        """
        if self.roth:
            # obtain cur_coefs (sigmoid self.coefs; zero out the out-of-scope coefs with scope_mask)
            cur_coefs = self.scopes_mask[...,None] * torch.sigmoid(self.coefs)
            cur_coefs_bar = self.scopes_mask[...,None] * torch.sigmoid(self.coefs_bar)
            # obtain cur_x (zero out the out-of-scope values with scope_mask)
            cur_x = self.scopes_mask[:,None,:] * torch.cat((torch.ones((x.size(0),1)), x), 1)[None,...]
            cur_x_bar = self.scopes_mask[:,None,:] * torch.cat((torch.ones((x.size(0),1)), 1-x), 1)[None,...]
            # compute roth polynomial (coefs dot x, normalize, log) torchishly
            res = torch.einsum("nbc,nck->bkn",cur_x,cur_coefs) + torch.einsum("nbc,nck->bkn",cur_x_bar,cur_coefs_bar)
            # compute normalizing constants torchishly
            Z = torch.transpose(2**(self.scope_sizes)[:,None] * cur_coefs[:,0,:] + 2**(self.scope_sizes-1)[:,None] * torch.sum(cur_coefs[:,1:,:],1), 0, 1)
            Z_bar = torch.transpose(2**(self.scope_sizes)[:,None] * cur_coefs_bar[:,0,:] + 2**(self.scope_sizes-1)[:,None] * torch.sum(cur_coefs_bar[:,1:,:],1), 0, 1)
            Z_total = Z + Z_bar
            self.prob = torch.log(res / Z_total[None,:,:])
            ############################################################################
            # BEFORE ADDING THE MINUS VERSION
            # # obtain cur_coefs (sigmoid self.coefs; zero out the out-of-scope coefs with scope_mask)
            # cur_coefs = self.scopes_mask[...,None] * torch.sigmoid(self.coefs)
            # # obtain cur_x (zero out the out-of-scope values with scope_mask)
            # cur_x = self.scopes_mask[:,None,:] * torch.cat((torch.ones((x.size(0),1)), x), 1)[None,...]
            # # compute roth polynomial (coefs dot x, normalize, log) torchishly
            # res = torch.einsum("nbc,nck->bkn",cur_x,cur_coefs)
            # # compute normalizing constants torchishly
            # Z = torch.transpose(2**(self.scope_sizes)[:,None] * cur_coefs[:,0,:] + 2**(self.scope_sizes)[:,None] * torch.sum(cur_coefs[:,1:,:],1), 0, 1)
            # self.prob = torch.log(res / Z[None,:,:])
        else:
            # ef_array has shape (batch, num_var, num_dist, num_repetition) 
            self.prob = torch.einsum('bxir,xro->bio', self.ef_array(x), self.scope_tensor)

    def backtrack(self, dist_idx, node_idx, mode='sample', **kwargs):
        """
        Backtrackng mechanism for EiNets.

        :param dist_idx: list of N indices into the distribution vectors, which shall be sampled.
        :param node_idx: list of N indices into the leaves, which shall be sampled.
        :param mode: 'sample' or 'argmax'; for sampling or MPE approximation, respectively.
        :param kwargs: keyword arguments
        :return: samples (Tensor). Of shape (N, self.num_var, self.num_dims).
        """
        if len(dist_idx) != len(node_idx):
            raise AssertionError("Invalid input.")

        with torch.no_grad():
            N = len(dist_idx)
            if mode == 'sample':
                ef_values = self.ef_array.sample(N, **kwargs)
            elif mode == 'argmax':
                ef_values = self.ef_array.argmax(**kwargs)
            else:
                raise AssertionError('Unknown backtracking mode {}'.format(mode))

            values = torch.zeros((N, self.num_var, self.num_dims), device=ef_values.device, dtype=ef_values.dtype)

            for n in range(N):
                cur_value = torch.zeros(self.num_var, self.num_dims, device=ef_values.device, dtype=ef_values.dtype)
                if len(dist_idx[n]) != len(node_idx[n]):
                    raise AssertionError("Invalid input.")
                for c, k in enumerate(node_idx[n]):
                    scope = list(self.nodes[k].scope)
                    rep = self.nodes[k].einet_address.replica_idx
                    if mode == 'sample':
                        cur_value[scope, :] = ef_values[n, scope, :, dist_idx[n][c], rep]
                    elif mode == 'argmax':
                        cur_value[scope, :] = ef_values[scope, :, dist_idx[n][c], rep]
                    else:
                        raise AssertionError('Unknown backtracking mode {}'.format(mode))
                values[n, :, :] = cur_value

            return values

    def em_set_hyperparams(self, online_em_frequency, online_em_stepsize, purge=True):
        self.ef_array.em_set_hyperparams(online_em_frequency, online_em_stepsize, purge)

    def em_purge(self):
        self.ef_array.em_purge()

    def em_process_batch(self):
        self.ef_array.em_process_batch()

    def em_update(self):
        self.ef_array.em_update()

    def project_params(self, params):
        self.ef_array.project_params(params)

    def reparam(self, params):
        return self.ef_array.reparam(params)

    # --------------------------------------------------------------------------------

    def set_marginalization_idx(self, idx):
        """Set indicices of marginalized variables."""
        self.ef_array.set_marginalization_idx(idx)

    def get_marginalization_idx(self):
        """Get indicices of marginalized variables."""
        return self.ef_array.get_marginalization_idx()





"""
OLD CODE ITERATIONS FOR THE ROTH-POLYNOMIAL LOG-DENSITY COMPUTATION (I.E. FORWARD)
            #print('self.prob', self.prob.size())
            #print(self.prob)
            #exit()
            #print(self.prob[:,:,c])
            #old: #torch.log(torch.dot(cur_x, cur_coefs, 1) / (2**(sc) * cur_coefs[:,0] + 2**(sc-1) * sum(cur_coefs[:,1:],1)))
            #exit()
            #######################################################################################################
            # BEFORE REMOVING THE LEAF-LOOP (C, NODE)
            # # compute log densities of shape (batch, K_num_dist, num_nodes)
            # self.prob = torch.zeros((x.size(0), self.num_dist, len(self.nodes)))
            # for c, node in enumerate(self.nodes):#self.nodes is specifically the leaves
            #     # create a mask for the scope of this leaf
            #     scope_mask = torch.cat((torch.Tensor([1.0]),torch.zeros(self.num_var).scatter(0,torch.LongTensor(list(node.scope)), 1)))
            #     # obtain coefs (sigmoid self.coefs; zero out the out-of-scope coefs with scope_mask)
            #     # (this is 'broadcasting' to make the multiplication work across the k-dimension)
            #     cur_coefs = scope_mask * torch.sigmoid(self.coefs[c,:,:])
            #     # obtain cur_x (zero out the out-of-scope values with scope_mask)
            #     cur_x =  scope_mask * torch.cat((torch.ones((x.size(0),1)), x), 1)
            #     # compute roth polynomial (dot product of coefs with x); normalize; take log
            #     # perform dot products manually torchishly
            #     #print(cur_x.size())
            #     #print(cur_coefs.size())
            #     res = torch.matmul(cur_x, torch.transpose(cur_coefs, 0, 1))
            #     #print(res.size())
            #     #before#res = torch.sum(cur_x * cur_coefs,1)
            #     #print('res',res.size())
            #     # compute normalizing constants torchishly
            #     Z = (2**(len(node.scope)) * cur_coefs[:,0] + 2**(len(node.scope)-1) * torch.sum(cur_coefs[:,1:],1))
            #     #print('Z', Z.size())
            #     self.prob[:,:,c] = torch.log(res / Z)
            #     #print(self.prob[:,:,c])
            #     #old: #torch.log(torch.dot(cur_x, cur_coefs, 1) / (2**(sc) * cur_coefs[:,0] + 2**(sc-1) * sum(cur_coefs[:,1:],1)))
            #     #exit()
            ##############################################################
            # BEFORE I REMOVED B-LOOP
            # # compute log densities of shape (batch, K_num_dist, num_nodes)
            # self.prob = torch.zeros((x.size(0), self.num_dist, len(self.nodes)))
            # for b in range(x.size(0)):
            #     for c, node in enumerate(self.nodes):#self.nodes is specifically the leaves
            #         # create a mask for the scope of this leaf
            #         scope_mask = torch.cat((torch.Tensor([1.0]),torch.zeros(self.num_var).scatter(0,torch.LongTensor(list(node.scope)), 1)))
            #         # obtain coefs (sigmoid self.coefs; zero out the out-of-scope coefs with scope_mask)
            #         # (this is 'broadcasting' to make the multiplication work across the k-dimension)
            #         cur_coefs = scope_mask * torch.sigmoid(self.coefs[c,:,:])
            #         # obtain cur_x (zero out the out-of-scope values with scope_mask)
            #         cur_x =  scope_mask * torch.cat((torch.tensor([1.0]), x[b]))
            #         # compute roth polynomial (dot product of coefs with x); normalize; take log
            #         # perform dot products manually torchishly
            #         res = torch.sum(cur_x * cur_coefs,1)
            #         #print('res',res.size())
            #         # compute normalizing constants torchishly
            #         Z = (2**(len(node.scope)) * cur_coefs[:,0] + 2**(len(node.scope)-1) * torch.sum(cur_coefs[:,1:],1))
            #         #print('Z', Z.size())
            #         self.prob[b,:,c] = res / Z
            #         #print(self.prob[b,:,c])
            #         #old: #torch.log(torch.dot(cur_x, cur_coefs, 1) / (2**(sc) * cur_coefs[:,0] + 2**(sc-1) * sum(cur_coefs[:,1:],1)))
            #         exit()
            ########################################################################
            # BEFORE I REMOVED K-LOOP
            # # compute log densities of shape (batch, K_num_dist, num_nodes)
            # self.prob = torch.zeros((x.size(0), self.num_dist, len(self.nodes)))
            # for b in range(x.size(0)):
            #     for k in range(self.num_dist):
            #         for c, node in enumerate(self.nodes):#self.nodes is specifically the leaves
            #             # create a mask for the scope of this leaf
            #             scope_mask = torch.cat((torch.Tensor([1.0]),torch.zeros(self.num_var).scatter(0,torch.LongTensor(list(node.scope)), 1)))
            #             # obtain coefs (sigmoid self.coefs; zero out the out-of-scope coefs with scope_mask)
            #             cur_coefs = scope_mask * torch.sigmoid(self.coefs[c,k,:])
            #             # obtain cur_x (zero out the out-of-scope values with scope_mask)
            #             cur_x =  scope_mask * torch.cat((torch.tensor([1.0]), x[b]))
            #             # compute roth polynomial (dot product of coefs with x); normalize; take log
            #             self.prob[b][k][c] = torch.log(torch.dot(cur_x, cur_coefs) / (2**(len(node.scope)) * cur_coefs[0] + 2**(len(node.scope)-1) * sum(cur_coefs[1:])))
            ########################################################################
            # VERSION WITH JAGGED (BUT MUCH SMALLER) COEFS MATRIX
            # self.prob = torch.zeros((x.size(0), self.num_dist, len(self.nodes)))
            # for b in range(x.size(0)):
            #     for k in range(self.num_dist):
            #         for c, node in enumerate(self.nodes):#self.nodes is specifically the leaves
            #             cur_coefs = torch.sigmoid(self.coefs[c])
            #             cur_x = torch.cat((torch.tensor([1.0]), torch.index_select(x[b], 0, torch.LongTensor(node.scope))))
            #             # compute roth polynomial (dot product of coefs with x) and normalize and then take log
            #             self.prob[b][k][c] = torch.log(torch.dot(cur_x, cur_coefs) / (2**(len(node.scope)) * cur_coefs[0] + 2**(len(node.scope)-1) * sum(cur_coefs[1:])))
            ######################################################################



OLD CODE FOR THE SELF.COEFS CREATION
            # self.coefs = []
            # for c, node in enumerate(self.nodes):
            #     # constant coefficient and coefficients for the variables in scope
            #     tmp = torch.randn((1+len(node.scope)))
            #     self.coefs.append(torch.nn.Parameter(tmp, requires_grad=True
"""