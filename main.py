import csv
import torch
import warnings
import argparse
import numpy as np
import torch.nn as nn
warnings.filterwarnings("ignore")

import os
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from utils import *
from GeneVAE import GeneVAE
from SmilesVAE import EncoderRNN, DecoderRNN, SmilesVAE
from train_gene_vae import train_gene_vae, load_test_gene_data
from train_smiles_vae import load_smiles_data, train_smiles_vae

# =================================================================================
# Default settings
parser = argparse.ArgumentParser()
parser.add_argument('--use_seed', action='store_true', help='Apply seed for reproduce experimental results') # Add --use_seed for reproduction
parser.add_argument('--cell_name', type=str, default='mcf7', help='Cell name of LINCS files, e.g., mcf7')
parser.add_argument('--protein_name', type=str, default='AKT1', help='10 proteins are AKT1, AKT2, AURKB, CTSK, EGFR, HDAC1, MTOR, PIK3CA, SMAD3, and TP53')

# ===========================
# GeneVAE
parser.add_argument('--train_gene_vae', action='store_true', help='Train GeneVAE') # Add --train_gene_vae to train GeneVAE
parser.add_argument('--test_gene_vae', action='store_true', help='Validate GeneVAE') # Add --test_gene_vae to test GeneVAE
parser.add_argument('--generation', action='store_true', help='Validate GeneVAE') # Add --test_gene_vae to test GeneVAE

parser.add_argument('--gene_epochs', type=int, default=2000, help='GeneVAE training epochs') 
parser.add_argument('--gene_num', type=int, default=978, help='Number of gene values') # MCF7: 978
parser.add_argument('--gene_hidden_sizes', type=int, default=[512, 256, 128], help='Hidden layer sizes of GeneVAE') # MCF7: [512, 256, 128, 100]
parser.add_argument('--gene_latent_size', type=int, default=64, help='Latent vector dimension of GeneVAE') # MCF7: 64
parser.add_argument('--gene_lr', type=float, default=1e-4, help='Learning rate of GeneVAE') # MCF7: 1e-4
parser.add_argument('--gene_batch_size', type=int, default=64, help='Batch size for training GeneVAE') # 64
parser.add_argument('--gene_dropout', type=float, default=0.2, help='Dropout probability')
parser.add_argument('--gene_expression_file', type=str, default='datasets/LINCS/', help='Path of the training gene expression profile dataset for the VAE')
parser.add_argument('--test_gene_data', type=str, default='datasets/test_protein/', help='Path of the gene expression profile dataset for test proteins or test disease')
parser.add_argument('--saved_gene_vae', type=str, default='results/saved_gene_vae', help='Save the trained GeneVAE')
parser.add_argument('--gene_vae_train_results', type=str, default='results/gene_vae_train_results.csv', help='Path to save the results of trained GeneVAE')
parser.add_argument('--one_gene_density_figure', type=str, default='results/one_gene_density_figure.png', help='Path to save the density figures of gene data')
parser.add_argument('--all_gene_density_figure', type=str, default='results/all_gene_density_figure.png', help='Path to save the density figures of gene data')

# ===========================
# SmilesVAE
parser.add_argument('--train_smiles_vae', action='store_true', help='Train SmilesVAE') # Add --train_smiles_vae to train SmilesVAE
parser.add_argument('--test_smiles_vae', action='store_true', help='Test SmilesVAE') # Add --test_smiles_vae to test SmilesVAE

parser.add_argument('--smiles_epochs', type=int, default=200, help='SmilesVAE training epochs')
parser.add_argument('--emb_size', type=int, default=128, help='Embedding size of SmilesVAE')
parser.add_argument('--hidden_size', type=int, default=256, help='Hidden layer size of SmilesVAE')
parser.add_argument('--num_layers', type=int, default=3, help='Number of layers for training SmilesVAE')
parser.add_argument('--smiles_latent_size', type=int, default=64, help='Latent vector dimension of SmilesVAE') # MCF7: 64
parser.add_argument('--smiles_lr', type=float, default=5e-4, help='Learning rate of SmilesVAE')
parser.add_argument('--bidirectional', type=bool, default='True', help='Apply bidirectional RNN')
parser.add_argument('--smiles_dropout', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--temperature', type=float, default=1.0, help='Temperature of the SMILES VAE')
parser.add_argument('--train_rate', type=float, default=0.9, help='Split training and validating subsets by training rate')
parser.add_argument('--max_len', type=int, default=100, help='Maximum length of SMILES strings')
parser.add_argument('--saved_smiles_vae', type=str, default='results/saved_smiles_vae.pkl', help='Save the trained SmilesVAE')
parser.add_argument('--valid_smiles_file', type=str, default='results/predicted_valid_smiles.csv', help='Save the valid SMILES into file')
parser.add_argument('--smiles_vae_train_results', type=str, default='results/smiles_vae_train_results.csv', help='Path to save the results of trained SmilesVAE')
parser.add_argument('--variant', action='store_true', help='Apply variant smiles') # Add --variant to apply variant smiles

# ===========================
# Molecule selection with similar ligands
parser.add_argument('--calculate_tanimoto', action='store_true', help='Calculate tanimoto similarity for the source ligand and generated SMILES') # Add --calculate_tanimoto to calculate Tanimoto similarity
parser.add_argument('--candidate_num', type=int, default=50, help='Number of candidate SMILES strings')

parser.add_argument('--gene_type', type=str, default='gene_symbol', help='Gene types')
parser.add_argument('--source_path', type=str, default='datasets/ligands/source_', help='Load the source SMILES strings of known ligands')
parser.add_argument('--gen_path', type=str, default='results/generation/', help='Save the generated SMILES strings')

args = parser.parse_args()


# =================================================================================
def main(args):

    if args.use_seed:
        # Apply the seed to reproduct the results
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

    # ========================================================= #
    #                    1. Train GeneVAE                       #
    # ========================================================= #
    if args.train_gene_vae:
        # Print GeneVAE hyperparameter information
        show_gene_vae_hyperparamaters(args)

        # Train GeneVAE for representation learning
        trained_gene_vae = train_gene_vae(args)

    else:
        # Load the trained GeneVAE  
        trained_gene_vae = GeneVAE(
            input_size=args.gene_num, 
            hidden_sizes=args.gene_hidden_sizes,
            latent_size=args.gene_latent_size,
            output_size=args.gene_num,
            activation_fn=nn.ReLU(),
            dropout=args.gene_dropout
        ).to(get_device())
        trained_gene_vae.load_model(args.saved_gene_vae + '_' + args.cell_name + '.pkl')
        print('Load the trained GeneVAE.')
    

    # ========================================================= #
    #                  2. Test GeneVAE                          #
    # ========================================================= #
    if args.test_gene_vae:
        # Print SmilesVAE hyperparameter information
        show_smiles_vae_hyperparamaters(args)

        # Compare real and reconstructed genes 
        show_all_gene_densities(args, trained_gene_vae)
        print('Gene expression profile distribution is created.')
    
    # Print vocabulary information
    tokenizer = vocabulary(args)
    tokenizer.build_vocab()
    # Get train and valid dataloader
    train_dataloder, valid_dataloder = load_smiles_data(tokenizer, args)


    # ========================================================= #
    #                 3. Train SmilesVAE                        #
    # ========================================================= #
    if args.train_smiles_vae:
        # Train SmilesVAE
        trained_smiles_vae = train_smiles_vae(
            trained_gene_vae,
            train_dataloder, 
            valid_dataloder,
            tokenizer, 
            args
        ).to(get_device())

    # ========================================================= #
    #           4. GxVAEs Generation                            #
    # ========================================================= #
    if args.generation:
        # Print other hyperparameter information
        show_other_hyperparamaters(args)

        # Load the trained SmilesVAE  
        trained_encoder = EncoderRNN(
            args.emb_size, 
            args.hidden_size, 
            args.num_layers, 
            args.smiles_latent_size, 
            args.bidirectional, 
            tokenizer
        ).to(get_device())

        trained_decoder = DecoderRNN(
            args.emb_size, 
            args.hidden_size, 
            args.num_layers, 
            args.smiles_latent_size, 
            args.gene_latent_size, # condition_size = gene_latent_size
            tokenizer
        ).to(get_device())
    
        trained_smiles_vae = SmilesVAE(
            trained_encoder, 
            trained_decoder
        ).to(get_device())

        trained_smiles_vae.load_model(args.saved_smiles_vae)

        # Test mode
        trained_gene_vae.eval()
        trained_smiles_vae.eval()

        # Load testing data
        test_gene_loader = load_test_gene_data(args)

        res_smiles = []

        for _, genes in enumerate(test_gene_loader):
            # Extract Gx as condition
            genes = genes.to(get_device())
            gene_latent_vectors, _ = trained_gene_vae(genes)
            # Sampling z
            if genes.size(0) != 1:
                rand_z = torch.randn(genes.size(0), args.smiles_latent_size).to(get_device()) # [batch_size, latent_size]
            else:
                rand_z = torch.randn(args.candidate_num, args.smiles_latent_size).to(get_device()) # [candidate_num, latent_size]
                gene_latent_vectors = gene_latent_vectors.repeat(args.candidate_num, 1)

            dec_sampled_char = trained_smiles_vae.generation(
                rand_z, 
                gene_latent_vectors, 
                args.max_len, 
                tokenizer
            )
            output_smiles = ["".join(tokenizer.decode(\
                dec_sampled_char[i].squeeze().detach().cpu().numpy()
                )).strip("^$ ") for i in range(dec_sampled_char.size(0))]
            res_smiles.append(output_smiles)

        test_data = pd.DataFrame(columns=['SMILES'], data=res_smiles[0]) # res_smiles is [[Smiles strings]], [0] to a list
        if not os.path.exists(args.gen_path):
            os.makedirs(args.gen_path)
        test_data.to_csv(args.gen_path + 'res-{}.csv'.format(args.protein_name), index=False)

    # ========================================================= #
    #              5. Tanimoto Calculation                      #
    # ========================================================= #
    if args.calculate_tanimoto:     
        # Read training data
        train_data = pd.read_csv(
            args.gene_expression_file + args.cell_name + '.csv', 
            sep=',', 
            names=['inchikey','smiles'] + ['gene'+str(i) for i in range(1,args.gene_num+1)]
        )
        train_data = train_data['smiles']
        # Canonical SMILES
        train_data = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in train_data]

        # Read source ligands
        source_path = args.source_path + args.protein_name + '.csv'
        source_data = pd.read_csv(source_path, names=['smiles'])
        # Remove the SMILES string from the source ligands that are the same as the training dataset
        canonical_source_data = []

        for smi in source_data['smiles']:
            try:
                cano_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
            except Exception:
                cano_smi = smi
            if cano_smi not in train_data:
                canonical_source_data.append(cano_smi)

        # Read generated SMILES string according to a protein
        if not os.path.exists(args.gen_path):
            print('generated path {} does not exist!'.format(args.gen_path))
        else:
            gen_path = args.gen_path + 'res-' + args.protein_name + '.csv'
            gen_data = pd.read_csv(gen_path)
            
            # Tanimoto similarity
            tanimoto = []
            valid_smiles = []

            for i in range(len(gen_data['SMILES'])):
                m1= Chem.MolFromSmiles(gen_data['SMILES'][i])
                if m1:
                    valid_smiles.append(Chem.MolToSmiles(m1))
                    try:
                        fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
                    except Exception:
                        break
                    else:
                        for j in range(len(canonical_source_data)):
                            try:
                                m2 = Chem.MolFromSmiles(canonical_source_data[j])
                                fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)
                            except Exception:
                                tanimoto.append([0, canonical_source_data[j], gen_data['SMILES'][i]])
                            else:
                                tanimoto.append([DataStructs.BulkTanimotoSimilarity(fp1, [fp2])[0], canonical_source_data[j], gen_data['SMILES'][i]])

            res = pd.DataFrame(tanimoto)   
            max_res = res.iloc[res[0].idxmax()]
            print('protein name:', args.protein_name)
            print('Source ligand:', max_res[1])
            print('Best generation:', max_res[2])
            print('Tanimoto similarity: {:.2f}'.format(max_res[0]))

            if len(valid_smiles) != 0:
                valid_rate = 100 * len(valid_smiles) / args.candidate_num
                
                unique_smiles = list(set(valid_smiles))
                unique_rate = 100 * len(unique_smiles) / len(valid_smiles)

                novel_smiles = [smi for smi in unique_smiles if smi not in train_data]
                novel_rate = 100 * len(novel_smiles) / len(unique_smiles)
            print('\n')
            print('Valid generation:', valid_rate)
            print('Unique generation:', unique_rate)
            print('Novel generation:', novel_rate)



if __name__ == '__main__':
    main(args)


