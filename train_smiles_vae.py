import math
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
from rdkit import Chem
from rdkit import rdBase

from GeneVAE import GeneVAE
from utils import Tokenizer, get_device, mean_similarity
from SmilesVAE import Smiles_DataLoader, EncoderRNN, DecoderRNN, SmilesVAE
rdBase.DisableLog('rdApp.error')

# ============================================================================
# Load data
def load_smiles_data(tokenizer, args):

    # Load smiles and gene values
    smiles_loader = Smiles_DataLoader(
        args.gene_expression_file, 
        args.cell_name,
        tokenizer,
        args.gene_num,
        batch_size=args.gene_batch_size,
        train_rate=args.train_rate,
        variant=args.variant
    )
    train_dataloader, valid_dataloader = smiles_loader.get_dataloader()

    return train_dataloader, valid_dataloader

# ============================================================================
# Generate Smiles using learned gene representations
def train_smiles_vae(
    trained_gene_vae,
    train_dataloader, 
    valid_dataloader,
    tokenizer,
    args
):
    """
    trained_gene_vae: the pretrained GeneVAE model for gene feature extraction
    train_dataloader: splited training data (encoded smiles, genes)
    tokenizer: SMILES vocabulary
    """
    # Define EncoderRNN
    encoder = EncoderRNN(
        args.emb_size, 
        args.hidden_size, 
        args.num_layers, 
        args.smiles_latent_size, 
        args.bidirectional, 
        tokenizer
    ).to(get_device())

    # Define DecoderRNN
    decoder = DecoderRNN(
        args.emb_size, 
        args.hidden_size, 
        args.num_layers, 
        args.smiles_latent_size, 
        args.gene_latent_size, # condition_size = gene_latent_size
        tokenizer
    ).to(get_device())
    
    # Define SmilesVAE
    smiles_vae = SmilesVAE(encoder, decoder).to(get_device())
    # Optimizer
    optimizer = torch.optim.Adam(
        smiles_vae.parameters(), 
        lr=args.smiles_lr
    )

    # Prepare file to save results
    with open(args.smiles_vae_train_results, 'a+') as wf:
        wf.truncate(0)
        wf.write('{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(\
            'Epoch', 
            'Joint_loss', 
            'Rec_loss',
            'KLD_loss',
            'Total', 
            'Valid', 
            'Valid_rate', 
            'Unique', 
            'Unique_rate', 
            'Novelty',
            'Novel_rate',
            'Diversity'
        ))

    print('\n')
    print('Training Information:')

    for epoch in range(args.smiles_epochs):

        total_joint_loss = 0
        total_rec_loss = 0
        total_kld_loss = 0
        smiles_vae.train()

        # Operate on a batch of data
        for _, (smiles, genes) in enumerate(train_dataloader):

            smiles, genes = smiles.to(get_device()), genes.to(get_device())

            # Extract gene expression features
            trained_gene_vae.eval()
            gene_latent_vectors, _ = trained_gene_vae(genes) # [batch_size, gene_latent_size]

            # Apply SmilesVAE (MolVAE)
            z, decoded = smiles_vae(smiles, gene_latent_vectors, args.temperature)
            alphas = torch.cat([
                torch.linspace(0.99, 0.5, int(args.smiles_epochs/2)), 
                0.5 * torch.ones(args.smiles_epochs - int(args.smiles_epochs/2))
            ]).double().to(get_device())

            joint_loss, rec_loss, kld_loss = smiles_vae.joint_loss(
                decoded, 
                targets=smiles,
                alpha=alphas[epoch],
                beta=1.
            )
            optimizer.zero_grad()
            joint_loss.backward()
            optimizer.step()

            total_joint_loss += joint_loss.item()
            total_rec_loss += rec_loss.item()
            total_kld_loss += kld_loss.item()

        mean_joint_loss = total_joint_loss / smiles.size(0)
        mean_rec_loss = total_rec_loss / (smiles.size(0))
        mean_kld_loss = total_kld_loss / (smiles.size(0))
                
        # Evaluate valid and unique SMILES
        smiles_vae.eval()
        valid_smiles = []
        label_smiles = []
        total_num_data = len(valid_dataloader.dataset)

        for _, (smiles, genes) in enumerate(valid_dataloader):

            smiles, genes = smiles.to(get_device()), genes.to(get_device())
            trained_gene_vae.eval()
            # Extracted Gx as condition
            gene_latent_vectors, _ = trained_gene_vae(genes)
            # Random values as smiles_z
            rand_z = torch.randn(genes.size(0), args.smiles_latent_size).to(get_device())
            dec_sampled_char = smiles_vae.generation(rand_z, gene_latent_vectors, args.max_len, tokenizer)
            output_smiles = ["".join(tokenizer.decode(\
                dec_sampled_char[i].squeeze().detach().cpu().numpy()
                )).strip("^$ ") for i in range(dec_sampled_char.size(0))]

            #valid_smiles.extend([smi for smi in output_smiles if Chem.MolFromSmiles(smi) and Chem.MolFromSmiles(smi)!=])
            for i in range(len(output_smiles)):

                mol = Chem.MolFromSmiles(output_smiles[i])
                if mol != None and mol.GetNumAtoms() > 1 and Chem.MolToSmiles(mol) != ' ':
                    valid_smiles.extend([output_smiles[i]])
                    label_smiles.extend(["".join(tokenizer.decode(smiles[i].squeeze().detach().cpu().numpy())).strip("^$ ")])

        unique_smiles = list(set(valid_smiles))
        # Novel Smiles
        novel_smiles = [smi for smi in unique_smiles if smi not in label_smiles]
        # Save valid Smiles to file
        valid_csv = pd.DataFrame(valid_smiles).to_csv(args.valid_smiles_file, index=False)
        
        valid_num = len(valid_smiles)
        valid_rate = 100*len(valid_smiles)/total_num_data
        unique_num = len(unique_smiles)
        novel_num = len(novel_smiles)

        if valid_num != 0:
            unique_rate = 100*unique_num/valid_num
            diversity = 1 - mean_similarity(valid_smiles, label_smiles)
        else:
            unique_rate = 100*unique_num/(valid_num+1)
            diversity = 1

        if unique_num != 0:
            novel_rate = 100*novel_num/unique_num
        else:
            novel_rate = 100*novel_num/(unique_num+1)
        
        print('Epoch: {:d} / {:d}, joint_loss: {:.3f}, rec_loss: {:.3f}, kld_loss: {:.3f}, Total: {:d}, valid: {:d} ({:.2f}), unique: {:d} ({:.2f}), novel: {:d} ({:.2f}), diversity: {:.3f}'.format(\
            epoch+1, 
            args.smiles_epochs, 
            mean_joint_loss, 
            mean_rec_loss,
            mean_kld_loss,
            total_num_data, 
            valid_num, 
            valid_rate,
            unique_num,
            unique_rate,
            novel_num,
            novel_rate,
            diversity
        ))

        # Save trained results to file
        with open(args.smiles_vae_train_results, 'a+') as wf:
            wf.write('{},{:.3f},{:.3f},{:.3f},{},{},{:.2f},{},{:.2f},{},{:.2f},{:.2f}\n'.format(\
                epoch+1, 
                mean_joint_loss, 
                mean_rec_loss,
                mean_kld_loss,
                total_num_data, 
                valid_num, 
                valid_rate,
                unique_num,
                unique_rate,
                novel_num,
                novel_rate,
                diversity
            ))

        # Save predicted and label SMILES into file
        final_smiles = {'predict': valid_smiles, 'label': label_smiles}
        final_smiles = pd.DataFrame(final_smiles)
        # Save to file
        final_smiles.to_csv(args.valid_smiles_file, index=False)

    print('='*50)
    # Save the trained SmilesVAE
    smiles_vae.save_model(args.saved_smiles_vae)
    print('Trained SmilesVAE is saved in {}'.format(args.saved_smiles_vae))

    return smiles_vae



































