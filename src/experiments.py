import torch
from EinsumNetwork import Graph, EinsumNetwork
import datasets
import os
import math
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#print('device is', device)
# choose a specific gpu
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# OVERRIDE AND USE CPU
device = 'cpu'

# various methods of obtaining a distribution over the k>=1 variables in the scope of a leaf node
EINET = 0           # fully factorized
ROTH = 1            # c0+c1x1+c2x2+...+ckxk
ROTH_BAR = 2        # c0+c1x1+c2x2+...+ckxk + c0+c1(1-x1)+c2(1-x2)+...+ck(1-xk)
DETERMINANT = 3     # det[M]
method_names = ['EINET', 'ROTH', 'ROTH_BAR', 'DETERMINANT']

methods = [EINET, ROTH, ROTH_BAR]#, DETERMINANT]
learning_rates = [0.01]#[0.005, 0.01, 0.05]
leaf_scopes = [5]#[5, 10,15]
dataset_names = ['accidents']#, 'ad', 'baudio', 'bbc', 'bnetflix', 'book', 'c20ng', 'cr52', 'cwebkb', 'dna', 'jester', 'kdd', 'kosarek', 'moviereview', 'msnbc', 'msweb', 'nltcs', 'plants', 'pumsb_star', 'tmovie', 'tretail', 'voting']

max_num_epochs = 5
num_repetitions = 10
num_input_distributions = 20
num_sums = 20
batch_size = 32

results = torch.empty((len(dataset_names),len(methods),len(learning_rates),len(leaf_scopes),max_num_epochs,3))# final dimension is for train, test, val average log-likelihoods

for d, dataset in enumerate(dataset_names):
    train_x_orig, test_x_orig, valid_x_orig = datasets.load_debd(dataset, dtype='float32')

    train_x = train_x_orig
    test_x = test_x_orig
    valid_x = valid_x_orig

    #print('Before .to(device)')
    train_x = torch.from_numpy(train_x).to(torch.device(device))
    valid_x = torch.from_numpy(valid_x).to(torch.device(device))
    test_x = torch.from_numpy(test_x).to(torch.device(device))
    #print('After .to(device)')

    train_N, num_dims = train_x.shape
    valid_N = valid_x.shape[0]
    test_N = test_x.shape[0]

    num_var = train_x.shape[1]

    for m, method in enumerate(methods):
        for l, learning_rate in enumerate(learning_rates):
            for s, leaf_scope in enumerate(leaf_scopes):
                print('method',method_names[m],'; learning_rate',learning_rate,'; leaf_scope',leaf_scope,'; dataset',dataset)

                depth = round(math.log(num_var / leaf_scope, 2))

                graph = Graph.random_binary_trees(num_var=num_var, depth=depth, num_repetitions=num_repetitions)

                use_em = False # if use_em is false, then SGD is used

                args = EinsumNetwork.Args(
                    num_classes=1,
                    num_input_distributions=num_input_distributions,
                    exponential_family=EinsumNetwork.CategoricalArray,
                    exponential_family_args={'K': 2},
                    num_sums=num_sums,
                    num_var=train_x.shape[1],
                    use_em=use_em)

                einet = EinsumNetwork.EinsumNetwork(graph, args, method=method)
                einet.initialize()
                einet.to(device)

                optimizer = torch.optim.Adam(einet.parameters(), lr=learning_rate)

                if use_em:
                    print('why the heck are you using EM?')
                    for epoch_count in range(max_num_epochs):
                        train_ll = EinsumNetwork.eval_loglikelihood_batched(einet, train_x)
                        valid_ll = EinsumNetwork.eval_loglikelihood_batched(einet, valid_x)
                        test_ll = EinsumNetwork.eval_loglikelihood_batched(einet, test_x)
                        idx_batches = torch.randperm(train_N).split(batch_size)
                        for batch_count, idx in enumerate(idx_batches):
                            batch_x = train_x[idx, :]
                            outputs = einet(batch_x)
                            ll_sample = EinsumNetwork.log_likelihoods(outputs)
                            log_likelihood = ll_sample.sum()
                            objective = log_likelihood
                            objective.backward()
                            einet.em_process_batch()
                        einet.em_update()
                else:
                    for epoch_count in range(max_num_epochs):
                        # evaluate
                        train_ll = EinsumNetwork.eval_loglikelihood_batched(einet, train_x)
                        valid_ll = EinsumNetwork.eval_loglikelihood_batched(einet, valid_x)
                        test_ll = EinsumNetwork.eval_loglikelihood_batched(einet, test_x)

                        results[d,m,l,s,epoch_count,:] = torch.Tensor([train_ll / train_N, valid_ll / valid_N, test_ll / test_N])
                        #print(results)
                        print("[{}]   train LL {}   valid LL {}  test LL {}".format(epoch_count, train_ll / train_N, valid_ll / valid_N, test_ll / test_N))

                        # train
                        idx_batches = torch.randperm(train_N).split(batch_size)
                        for batch_count, idx in enumerate(idx_batches):
                            batch_x = train_x[idx, :]
                            optimizer.zero_grad()
                            outputs = einet(batch_x)
                            ll_sample = EinsumNetwork.log_likelihoods(outputs)
                            log_likelihood = ll_sample.sum()
                            objective = -log_likelihood
                            objective.backward()
                            optimizer.step()


data = {}
data["dataset_names"] = dataset_names
data["methods"] = methods
data["learning_rates"] = learning_rates
data["leaf_scopes"] = leaf_scopes
data["max_num_epochs"] = max_num_epochs
data["results"] = results

import datetime
time_str = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
torch.save(data,"data_"+time_str)
