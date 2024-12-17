import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from GeneVAE import GeneVAE
from utils import get_device, common

# ============================================================================
# Define gene expression dataset
class GeneExpressionDataset(torch.utils.data.Dataset):
    
    def __init__(self, data):

        self.data = data
        self.data_num = len(data)

    def __len__(self):

        return self.data_num

    def __getitem__(self, idx):
        gene_data = torch.tensor(self.data.iloc[idx]).float()
        
        return gene_data

# ============================================================================
# Load gene expression dataset
def load_gene_expression_dataset(args):
    
    # Load data, which contains smiles, inchikey, and gene values
    data = pd.read_csv(
        args.gene_expression_file + args.cell_name + '.csv', 
        sep=',', 
        names=['inchikey','smiles'] + ['gene'+str(i) for i in range(1,args.gene_num+1)]
    )
    # Use only gene values to train the GeneVAE (omit smiles and inchikey)
    data = data.iloc[:, 2:]
    # Drop the nan row
    data = data.dropna(how='any')
    # Normalize data per gene 
    #data = (data - data.mean())/data.std()

    # Get a batch of gene data
    train_data = GeneExpressionDataset(data)

    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=args.gene_batch_size, 
        shuffle=True
    )

    return train_loader

# ============================================================================
# Load testing gene expression dataset
def load_test_gene_data(args):

    # Load data, which contains gene values
    data = pd.read_csv(
        args.test_gene_data + args.protein_name + '.csv', 
        sep=',',
        names=['name'] + ['gene'+str(i) for i in range(1,args.gene_num+1)]
    )
    
    data = data.iloc[:,1:]
    # Common the gene data with the columns of the source gene expression profiles
    data = common(data, args.gene_type)
    # Get a batch of gene data
    test_data = GeneExpressionDataset(data)
    test_loader = torch.utils.data.DataLoader(
        test_data, 
        batch_size=args.gene_batch_size, 
        shuffle=False
    )

    return test_loader

# ============================================================================
# Train GeneVAE (ProfileVAE)
def train_gene_vae(args):
  
    # Load gene dataset
    train_loader = load_gene_expression_dataset(args)
    
    # Define GeneVAE 
    gene_vae = GeneVAE(
        input_size=args.gene_num, 
        hidden_sizes=args.gene_hidden_sizes,
        latent_size=args.gene_latent_size,
        output_size=args.gene_num,
        activation_fn=nn.ReLU(),
        dropout=args.gene_dropout
    ).to(get_device())

    # Optimizer
    gene_optimizer = optim.Adam(gene_vae.parameters(), lr=args.gene_lr)

    # Gradually decrease the alpha (weight of MSE relative to KL)
    alpha = 0.5
    alphas = torch.cat([
        torch.linspace(0.99, alpha, int(args.gene_epochs/2)), 
        alpha * torch.ones(args.gene_epochs - int(args.gene_epochs/2))
    ]).double().to(get_device())

    # Prepare file to save results
    with open(args.gene_vae_train_results, 'a+') as wf:
        wf.truncate(0)
        wf.write('{},{},{},{}\n'.format('Epoch', 'Joint', 'Rec', 'KLD'))

    print('Training Information:')
    for epoch in range(args.gene_epochs):

        total_joint_loss = 0
        total_rec_loss = 0
        total_kld_loss = 0
        gene_vae.train()
        
        for _, genes in enumerate(train_loader):

            genes = genes.to(get_device())
            _, rec_genes = gene_vae(genes)
            joint_loss, rec_loss, kld_loss = gene_vae.joint_loss(
                outputs=rec_genes, 
                targets=genes,
                alpha=alphas[epoch],
                beta=1.
            )

            gene_optimizer.zero_grad()
            joint_loss.backward()
            gene_optimizer.step()

            total_joint_loss += joint_loss.item()
            total_rec_loss += rec_loss.item()
            total_kld_loss += kld_loss.item()
        
        mean_joint_loss = total_joint_loss / len(train_loader.dataset)
        mean_rec_loss = total_rec_loss / (len(train_loader.dataset) * args.gene_num)
        mean_kld_loss = total_kld_loss / (len(train_loader.dataset) * args.gene_latent_size)
        print('Epoch {:d} / {:d}, joint_loss: {:.3f}, rec_loss: {:.3f}, kld_loss: {:.3f},'.format(\
            epoch+1, args.gene_epochs, mean_joint_loss, mean_rec_loss, mean_kld_loss))
        
        # Save trained results to file
        with open(args.gene_vae_train_results, 'a+') as wf:
            wf.write('{},{:.3f},{:.3f},{:.3f}\n'.format(epoch+1, mean_joint_loss, mean_rec_loss, mean_kld_loss))
        
    # Save trained GeneVAE
    gene_vae.save_model(args.saved_gene_vae + '_' + args.cell_name + '.pkl')
    print('Trained GeneVAE is saved in {}'.format(args.saved_gene_vae + '_' + args.cell_name + '.pkl'))

    return gene_vae



























