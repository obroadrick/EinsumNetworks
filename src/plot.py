import matplotlib.pyplot as plt
import json

fname = "results_bar.json"
with open(fname,'r') as results_file:
    results = json.load(results_file)

roth = results['roth']
einet = results['einet']

dataset_names = ['accidents', 'ad', 'baudio', 'bbc', 'bnetflix', 'book', 'c20ng', 'cr52', 'cwebkb', 'dna', 'jester', 'kdd', 'kosarek', 'moviereview', 'msnbc', 'msweb', 'nltcs', 'plants', 'pumsb_star', 'tmovie', 'tretail', 'voting']

print(fname)
print('DATASET', 'EINET TEST LL', 'ROTH TEST LL')
for dname in dataset_names:
    print(dname, round(einet[dname][-1][-1],2), round(roth[dname][-1][-1],2))


