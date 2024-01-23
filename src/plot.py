import matplotlib.pyplot as plt
import torch
import json

fname = "data_02:53PM_January_23_2024"
data = torch.load(fname)

dataset_names = data["dataset_names"]
methods = data["methods"]
learning_rates = data["learning_rates"]
leaf_scopes = data["leaf_scopes"]
max_num_epochs = data["max_num_epochs"]
results = data["results"]

for d, dataset in enumerate(dataset_names):
    for m, method in enumerate(methods):
        for l, learning_rate in enumerate(learning_rates):
            for s, leaf_scope in enumerate(leaf_scopes):
                for epoch_count in range(max_num_epochs):
                    print(results[d,m,l,s,epoch_count,:])






# fname = "results_bar.json"
# with open(fname,'r') as results_file:
#     results = json.load(results_file)

# roth = results['roth']
# einet = results['einet']

# dataset_names = ['accidents', 'ad', 'baudio', 'bbc', 'bnetflix', 'book', 'c20ng', 'cr52', 'cwebkb', 'dna', 'jester', 'kdd', 'kosarek', 'moviereview', 'msnbc', 'msweb', 'nltcs', 'plants', 'pumsb_star', 'tmovie', 'tretail', 'voting']

# print(fname)
# print('DATASET', 'EINET TEST LL', 'ROTH TEST LL')
# for dname in dataset_names:
#     print(dname, round(einet[dname][-1][-1],2), round(roth[dname][-1][-1],2))


