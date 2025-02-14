import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch_geometric.datasets import TUDataset

from tqdm import trange

from util import compute_PI_tensor, label_propagation, reg_scheduler
from models.tensorgcn import TenGCN

criterion = nn.CrossEntropyLoss()

def train(args, model, device, source_graphs, source_PIs, target_graphs, target_PIs, optimizer, epoch):
    model.train()

    pbar = trange(args.iters_per_epoch, unit='batch')
    pbar.set_description('epoch: %d' % (epoch))
    
    for pos in pbar:
        ## source
        selected_idx = np.random.permutation(len(source_graphs))[:args.batch_size]
        batch_graph_source = [source_graphs[idx] for idx in selected_idx]
        batch_PI_source = torch.stack([source_PIs[idx] for idx in selected_idx])
        output_source, output_source_gcn, output_source_top = model(batch_graph_source,batch_PI_source,0)
        # supervised loss
        labels_source = torch.LongTensor([graph.y for graph in batch_graph_source]).to(device)
        sup_loss = criterion(output_source, labels_source)
       
        ## target
        selected_idx = np.random.permutation(len(target_graphs))[:args.batch_size]
        batch_graph_target = [target_graphs[idx] for idx in selected_idx]
        batch_PI_target = torch.stack([target_PIs[idx] for idx in selected_idx])
        _, output_target_gcn, output_target_top = model(batch_graph_target,batch_PI_target,1)
        # label propagation
        lp_gcn = label_propagation(output_source_gcn, output_target_gcn, labels_source, args.threshold, criterion, device)
        lp_top = label_propagation(output_source_top, output_target_top, labels_source, args.threshold, criterion, device)
        
        ## BP
        reg_gcn, reg_top = reg_scheduler(args.reg_gcn, args.reg_top, pos, epoch, args.iters_per_epoch)
        loss = reg_gcn * lp_gcn + reg_top * lp_top + sup_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, PIs, minibatch_size=32):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        batch_graph = [graphs[j] for j in sampled_idx]
        batch_PI = torch.stack([PIs[j] for j in sampled_idx])
        batch_output, _, _ = model(batch_graph,batch_PI,1)
        output.append(batch_output.detach())
    return torch.cat(output, 0)

def test(model, device, target_graphs, target_PIs):
    model.eval()

    output = pass_data_iteratively(model, target_graphs, target_PIs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.y for graph in target_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(target_graphs))
    
    print("accuracy test: %f" % acc_test)
    return acc_test

def main():
    parser = argparse.ArgumentParser(description='LP-TGNN')
    parser.add_argument('--source_dataset', type=str, default="COX2",
                        help='name of source dataset (default: MUTAG)')
    parser.add_argument('--target_dataset', type=str, default="COX2_MD",
                        help='name of target dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of GCN layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='number of hidden units (default: 32)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--tensor_layer_type', type = str, default = "TCL", choices=["TCL","TRL"],
                                        help='Tensor layer type: TCL/TRL')
    parser.add_argument('--node_pooling', action="store_true",
    					help='node pooling based on node scores')
    parser.add_argument('--sublevel_filtration_methods', nargs='+', type=str, default=['degree','betweenness','eigenvector','closeness'],
    					help='Methods for sublevel filtration on PDs')
    parser.add_argument('--PI_dim', type=int, default=50,
                        help='PI size: PI_dim * PI_dim')
    parser.add_argument('--reg_gcn', type=float, default=0.5,
                        help='weight of GCN branch regularization')
    parser.add_argument('--reg_top', type=float, default=0.5,
                        help='weight of Top branch regularization')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='threshold for label propagation')
    args = parser.parse_args()

    random_seed = 0
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    source_graphs = TUDataset(root='/tmp/' + args.source_dataset, name=args.source_dataset)
    target_graphs = TUDataset(root='/tmp/' + args.target_dataset, name=args.target_dataset)
    assert source_graphs.num_classes == target_graphs.num_classes, 'NOT a valid domain adaptation task!'
    num_classes = source_graphs.num_classes
    ## NOTE: compute graph PI tensor if necessary
    # source_PIs = compute_PI_tensor(source_graphs,args.PI_dim,args.sublevel_filtration_methods)
    # torch.save(source_PIs,'PI/{}_PI.pt'.format(args.source_dataset))
    # target_PIs = compute_PI_tensor(target_graphs,args.PI_dim,args.sublevel_filtration_methods)
    # torch.save(target_PIs,'PI/{}_PI.pt'.format(args.target_dataset))
    ## load pre-computed PIs
    source_PIs = torch.load('PI/{}_PI.pt'.format(args.source_dataset)).to(device)
    print('finished loading PI for dataset {}'.format(args.source_dataset))
    target_PIs = torch.load('PI/{}_PI.pt'.format(args.target_dataset)).to(device)
    print('finished loading PI for dataset {}'.format(args.target_dataset))
    
    model = TenGCN(args.num_layers, args.num_mlp_layers, source_graphs[0].x.shape[1], target_graphs[0].x.shape[1], args.hidden_dim, num_classes, args.final_dropout, args.tensor_layer_type, args.node_pooling, args.PI_dim, args.sublevel_filtration_methods, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    max_acc = 0.0
    for epoch in range(args.epochs):
        print("Current epoch is:", epoch)

        train(args, model, device, source_graphs, source_PIs, target_graphs, target_PIs, optimizer, epoch)
        acc_test = test(model, device, target_graphs, target_PIs)
        max_acc = max(max_acc, acc_test)

    with open('res/' + str(args.source_dataset) + '_' + str(args.target_dataset) + '.txt', 'a+') as f:
        f.write(str(max_acc) + '\n')

if __name__ == '__main__':
    main()