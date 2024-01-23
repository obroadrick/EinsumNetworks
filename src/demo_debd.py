import torch
from EinsumNetwork import Graph, EinsumNetwork
import datasets
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device is', device)
# choose a specific gpu
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

### use cpu instead
device = 'cpu'
print('device is actually', device)
###

demo_text = """
This demo loads one of the 20 binary datasets and quickly trains an EiNet for some epochs. 

There are some parameters to play with, as for example which dataset shall be used and some 
structural parameters.
"""
print(demo_text)

##########################################################
dataset = 'accidents'

# depth = 3
# num_repetitions = 3
# num_input_distributions = 3
# num_sums = 3

# max_num_epochs = 3
# batch_size = 32
# online_em_frequency = 1
# online_em_stepsize = 0.05

# original settings:
depth = 3
num_repetitions = 10
num_input_distributions = 20
num_sums = 20

max_num_epochs = 5
batch_size = 100
online_em_frequency = 1
online_em_stepsize = 0.05

##########################################################

print(dataset)

train_x_orig, test_x_orig, valid_x_orig = datasets.load_debd(dataset, dtype='float32')
print('here1')

train_x = train_x_orig
test_x = test_x_orig
valid_x = valid_x_orig
print('here2')
# to torch
train_x = torch.from_numpy(train_x).to(device)
print('oh yeah')
valid_x = torch.from_numpy(valid_x).to(torch.device(device))
test_x = torch.from_numpy(test_x).to(torch.device(device))
print('here3')

train_N, num_dims = train_x.shape
valid_N = valid_x.shape[0]
test_N = test_x.shape[0]
print('here4')

graph = Graph.random_binary_trees(num_var=train_x.shape[1], depth=depth, num_repetitions=num_repetitions)
print('here5')

args = EinsumNetwork.Args(
    num_classes=1,
    num_input_distributions=num_input_distributions,
    exponential_family=EinsumNetwork.CategoricalArray,
    exponential_family_args={'K': 2},
    num_sums=num_sums,
    num_var=train_x.shape[1],
    use_em=False,
    online_em_frequency=1,
    online_em_stepsize=0.05)

einet = EinsumNetwork.EinsumNetwork(graph, args)
einet.initialize()
einet.to(device)
print(einet)


# newthings:
lr = 0.05
optimizer = torch.optim.Adam(einet.parameters(), lr=lr)


for epoch_count in range(max_num_epochs):

    # evaluate
    train_ll = EinsumNetwork.eval_loglikelihood_batched(einet, train_x)
    valid_ll = EinsumNetwork.eval_loglikelihood_batched(einet, valid_x)
    test_ll = EinsumNetwork.eval_loglikelihood_batched(einet, test_x)

    print("[{}]   train LL {}   valid LL {}  test LL {}".format(epoch_count,
                                                                train_ll / train_N,
                                                                valid_ll / valid_N,
                                                                test_ll / test_N))

    # train
    idx_batches = torch.randperm(train_N).split(batch_size)

    for batch_count, idx in enumerate(idx_batches):
        batch_x = train_x[idx, :]

        # new
        optimizer.zero_grad()

        outputs = einet(batch_x)

        ll_sample = EinsumNetwork.log_likelihoods(outputs)
        log_likelihood = ll_sample.sum()

        objective = -log_likelihood
        objective.backward()

        # new
        optimizer.step()

        #einet.em_process_batch()

    #einet.em_update()
