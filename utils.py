import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
from rdkit.DataStructs import FingerprintSimilarity

# ============================================================================
# KL Divergence loss 
def kld_loss(mu, logvar):
    """
    mu: Means of encoder output [batch_size, latent_size]
    logvar: log varances of encoder output [batch_size, latent_size]
    returns:
        KLD of the specified distribution and a unit Gaussian.
    """

    mu = mu.double().to(get_device())
    logvar = logvar.double().to(get_device())

    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return kld

# ============================================================================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
def show_gene_vae_hyperparamaters(args):

    # Hyper-parameters
    params = {}
    print('\n\nGeneVAE Hyperparameter Information:')
    print('='*50)
    params['GENE_EXPRESSION_FILE'] = args.gene_expression_file 
    params['CELL_NAME'] = args.cell_name
    params['GENE_EPOCHS'] = args.gene_epochs 
    params['GENE_LR'] = args.gene_lr
    params['GENE_NUM'] = args.gene_num
    params['GENE_HIDDEN_SIZES'] = args.gene_hidden_sizes
    params['GENE_LATENT_SIZE'] = args.gene_latent_size
    params['GENE_BATCH_SIZE'] = args.gene_batch_size
    params['GENE_DROUPOUT'] = args.gene_dropout

    for param in params:
        string = param + ' ' * (5 - len(param))
        print('{}:   {}'.format(string, params[param]))
    print('='*50)

# ============================================================================
def show_smiles_vae_hyperparamaters(args):

    # Hyper-parameters
    params = {}
    print('\n\nSmilesVAE Hyperparameter Information:')
    print('='*50)
    params['VALID_SMILES_FILE'] = args.valid_smiles_file
    params['SMILES_EPOCHS'] = args.smiles_epochs
    params['EMB_SIZE'] = args.emb_size
    params['HIDDEN_SIZE'] = args.hidden_size
    params['NUM_LAYERS'] = args.num_layers 
    params['SMILES_LR'] = args.smiles_lr
    params['SMILES_DROUPOUT'] = args.smiles_dropout
    params['TRAIN_RATE'] = args.train_rate

    for param in params:
        string = param + ' ' * (5 - len(param))
        print('{}:   {}'.format(string, params[param]))
    print('='*50)

# ============================================================================
def show_other_hyperparamaters(args):

    # Hyper-parameters
    params = {}
    print('\n\nOther Hyperparameter Information:')
    print('='*50)
    params['PROTEIN_NAME'] = args.protein_name
    params['SOURCE_PATH'] = args.source_path
    params['GEN_PATH'] = args.gen_path
    params['candidate_num'] = args.candidate_num

    for param in params:
        string = param + ' ' * (5 - len(param))
        print('{}:   {}'.format(string, params[param]))
    print('='*50)

# ============================================================================
# Build vocabulary for SMILES string data 
class Tokenizer():

    def __init__(self):
        self.start = "^"
        self.end = "$"
        self.pad = ' '
    
    def build_vocab(self):
        chars=[]
        # atoms 
        chars = chars + ['H', 'B', 'C', 'c', 'N', 'n', 'O', 'o', 'P', 'S', 's', 'F', 'I']
        # replace Si for Q, Cl for R, Br for V
        chars = chars + ['Q', 'R', 'V', 'Y', 'Z', 'G', 'T', 'U']
        # hidrogens: H2 to W, H3 to X
        chars = chars + ['[', ']', '+', 'W', 'X']
        # bounding
        chars = chars + ['-', '=', '#', '.', '/', '@', '\\']
        # branches
        chars = chars + ['(', ')']
        # cycles
        chars = chars + ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        #padding value is 0
        self.tokenlist = [self.pad, self.start, self.end] + list(chars)
        # create the dictionaries      
        self.char_to_int = {c:i for i,c in enumerate(self.tokenlist)}
        self.int_to_char = {i:c for c,i in self.char_to_int.items()}
    
    @property
    def vocab_size(self):
        return len(self.int_to_char)
    
    def encode(self, smi):
        encoded = []
        smi = smi.replace('Si', 'Q')
        smi = smi.replace('Cl', 'R')
        smi = smi.replace('Br', 'V')
        smi = smi.replace('Pt', 'Y')
        smi = smi.replace('Se', 'Z')
        smi = smi.replace('Li', 'T')
        smi = smi.replace('As', 'U')
        smi = smi.replace('Hg', 'G')
        # hydrogens
        smi = smi.replace('H2', 'W')
        smi = smi.replace('H3', 'X')

        return [self.char_to_int[self.start]] + [self.char_to_int[s] for s in smi] + [self.char_to_int[self.end]]
    
    def decode(self, ords):
        smi = ''.join([self.int_to_char[o] for o in ords]) 
        # hydrogens
        smi = smi.replace('W', 'H2')
        smi = smi.replace('X', 'H3')
        # replace proxy atoms for double char atoms symbols
        smi = smi.replace('Q', 'Si')
        smi = smi.replace('R', 'Cl')
        smi = smi.replace('V', 'Br')
        smi = smi.replace('Y', 'Pt')
        smi = smi.replace('Z', 'Se')
        smi = smi.replace('T', 'Li')
        smi = smi.replace('U', 'As')
        smi = smi.replace('G', 'Hg')
        
        return smi

    @property
    def n_tokens(self):
        return len(self.int_to_char)

# ============================================================================
def vocabulary(args):

    # Build the vocabulary
    tokenizer = Tokenizer()
    tokenizer.build_vocab()
    #print('\n')
    #print('Vocabulary Information:')
    #print('='*50)
    #print(tokenizer.char_to_int)
    #print('='*50)

    return tokenizer

# ============================================================================
def show_density(
    args, 
    figure_path, 
    row_num, 
    trained_gene_vae=None
):
    """
    figure_path: the path to save the figure
    row_num: number of rows of gene expression profile data used for data distribution
    """

    # Real gene expression profile data loading
    real_genes = pd.read_csv(
        args.gene_expression_file + args.cell_name + '.csv', 
        sep=',', 
        names=['inchikey','smiles'] + ['gene'+str(i) for i in range(1,args.gene_num+1)]
    )
    # Use only gene values to train the GeneVAE (omit smiles and inchikey)
    real_genes = real_genes.iloc[:, 2:]
    # Drop the nan row
    real_genes = real_genes.dropna(how='any')
    # Normalize data per gene
    #real_genes = (real_genes - real_genes.mean())/real_genes.std()

    # Calculate average value
    if row_num == 1:
        random_rows = np.array([1])
        #random_rows = np.random.choice(len(real_genes), row_num)
    else:
        random_rows = np.random.choice(len(real_genes), row_num)
    real_genes = real_genes.iloc[random_rows, :]
    mean_real_all_gene = real_genes.mean()

    plt.subplots(figsize=(12,7))
    plt.title("Data distribution of gene expression profile", fontsize=28)
    plt.xlabel("Values of gene expression profile data", fontsize=28)
    plt.ylabel("Density", fontsize=28)

    # Figure density distribution
    sns.histplot(mean_real_all_gene, bins=50, kde=True, label='Real gene', color='g')
    
    if trained_gene_vae:
        trained_gene_vae.eval()
        # Reconstructed gene
        inputs = torch.tensor(real_genes.values, dtype=torch.float32).to(get_device())
        _, rec_genes = trained_gene_vae(inputs)
        rec_genes = pd.DataFrame(rec_genes.cpu().detach().numpy())
        # Calculate average value
        mean_rec_gene = rec_genes.mean()
        # Figure density distribution
        sns.histplot(mean_rec_gene, bins=50, kde=True, label='Reconstructed gene', color='r')
    
    plt.legend()
    plt.savefig(figure_path, dpi=150)

def show_all_gene_densities(args, trained_gene_vae):

    show_density(args, args.one_gene_density_figure, 1, trained_gene_vae)
    show_density(args, args.all_gene_density_figure, 10000, trained_gene_vae)

# ============================================================================
def tanimoto_similarity(smi1, smi2):
    """
    smi1: SMILES string 1
    smi2: SMILES string 2
    returns:
        Tanimoto similarity score
    """
    mols = [Chem.MolFromSmiles(smi1), Chem.MolFromSmiles(smi2)]
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024) for mol in mols]
    sim_score = FingerprintSimilarity(fps[0],fps[1])

    return sim_score

def mean_similarity(pred_smiles, label_smiles):
    
    all_scores = [tanimoto_similarity(pred, label) for pred, label in zip(pred_smiles, label_smiles)]
    mean_score = np.mean(all_scores)

    return mean_score

# ============================================================================
def symbol2hsa(input_symbol):
    with open('datasets/tools/symbol2hsa.json', mode='rt', encoding='utf-8')as f:
        symbol_data = json.load(f)
        symbols = list(symbol_data.keys())
    hsas = []
    for sym in input_symbol:
        if sym in symbols:
            hsas.append(symbol_data[sym])
        else:
            hsas.append('-')
    return hsas

def common(df_tgt, gene_type):
    # Source gene names
    df_source = pd.read_csv('datasets/tools/source_genes.csv', sep=',')
    source_hsas = list(df_source.columns)
    # Target gene names
    tgt_hsas = list(df_tgt.columns)
    
    if not gene_type == 'gene_symbol':
        tgt_hsas = symbol2hsa(tgt_hsas)
        df_tgt = df_tgt.set_axis(tgt_hsas, axis=1)
   
    # Common gene names
    common_hsas = list(set(tgt_hsas) & set(source_hsas))
    common_hsas = sorted(common_hsas, key=source_hsas.index)
    # Processed target gene expression profile data
    df_source[common_hsas] = df_tgt[common_hsas]
    
    return df_source














