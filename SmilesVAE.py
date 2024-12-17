import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from rdkit import Chem
from GeneVAE import GeneVAE
from utils import get_device
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# ============================================================================
# Define the SMILES dataset
class Smiles_Dataset(Dataset):
    
    def __init__(
        self, 
        gene_expression_file, 
        cell_name,
        tokenizer,
        gene_num,
        variant
    ):
        """
        gene_expression_file: original gene data file
        cell_name: cell name, e.g., MCF7
        tokenizer: vocabulary to encode and decode SMILES
        gene_num: number of gene columns
        variant: True → Apply variant SMILES
        """
        data = pd.read_csv(
            gene_expression_file + cell_name + '.csv', 
            sep=',', 
            names=['inchikey','smiles']+['gene'+str(i) for i in range(1, gene_num+1)]
        )
        # Drop the nan row
        data = data.dropna(how='any')
        # Normalize data per gene
        #data.iloc[:, 2:] = (data.iloc[:, 2:] - data.iloc[:, 2:].mean())/data.iloc[:, 2:].std()
        
        if variant:
            # Variant SMILES
            data['smiles'] = data['smiles'].apply(self.variant_smiles)

        self.data = data

        self.tokenizer = tokenizer
    
    def __len__(self):

        return len(self.data)
    
    def __getitem__(self, index):

        smi = self.data.iloc[index]['smiles']
        # Encode SMILES strings
        encoded_smi = self.tokenizer.encode(smi)
        gene = self.data.iloc[index]['gene1':].values.astype('float32')
        
        return encoded_smi, gene

    def variant_smiles(self, smi):
        
        mol = Chem.MolFromSmiles(smi)
        atom_idxs = list(range(mol.GetNumAtoms()))
        np.random.shuffle(atom_idxs)
        mol = Chem.RenumberAtoms(mol,atom_idxs)

        return Chem.MolToSmiles(mol, canonical=False)

# ============================================================================
# Define the SMILES dataLoader
class Smiles_DataLoader(DataLoader):
    
    def __init__(
        self,
        gene_expression_file,
        cell_name,
        tokenizer,
        gene_num,
        batch_size,
        train_rate=0.9,
        variant=False
    ):
        """
        gene_expression_file: original gene data file
        cell_name: cell name, e.g., MCF7
        tokenizer: vocabulary to encode and decode SMILES
        gene_num: number of gene columns
        batch_size: batch size of gene data
        train_rate: split training and testing gene data by train rate
        variant: If true, apply variant SMILES
        """

        self.gene_expression_file = gene_expression_file
        self.cell_name = cell_name
        self.tokenizer = tokenizer
        self.gene_num = gene_num
        self.batch_size = batch_size
        self.train_rate = train_rate
        self.variant = variant
        
    def collate_fn(self, batch):

        # Batch is a list of zipped encoded smiles and genes
        smiles, genes = zip(*batch)
        smi_tensors = [torch.tensor(smi).squeeze(0) for smi in smiles]
        gene_tensors = torch.tensor(np.array(genes)) # [batch_size, gene_num]
        # Pad the different lengths of tensors to the maximum length
        smi_tensors = torch.nn.utils.rnn.pad_sequence(smi_tensors, batch_first=True) # [batch_size, max_len]
        
        return smi_tensors, gene_tensors
        
    def get_dataloader(self):

        # Load dataset
        dataset = Smiles_Dataset(
            self.gene_expression_file, 
            self.cell_name,
            self.tokenizer, 
            self.gene_num,
            self.variant
        )

        train_size = int(len(dataset) * self.train_rate)
        test_size = len(dataset) - train_size

        # Split train and test data
        train_data, test_data = random_split(
            dataset=dataset, 
            lengths=[train_size, test_size], 
            generator=torch.Generator().manual_seed(0)
        )

        train_dataloader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=1
        )

        test_dataloader = DataLoader(
            test_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=1
        )

        return train_dataloader, test_dataloader

# ============================================================================
# Define EncoderRNN: encode a batch of SMILES to Z (MolVAE)
class EncoderRNN(nn.Module):
    
    def __init__(
        self, 
        emb_size,
        hidden_size,
        num_layers,
        latent_size,
        bidirectional,
        tokenizer
    ):
        """
        args:
            - emb_size: embedding size for SMILES tokens
            - hidden_size: hidden layer size of RNN
            - num_layers: number of layers of RNN
            - latent_size: size of the latent vector
            - tokenizer: tokenizer of SMILES string dataset
        """
        super(EncoderRNN, self).__init__()
        
        self.emb_size = emb_size 
        self.hidden_size = hidden_size 
        self.latent_size = latent_size
        
        self.tokenizer = tokenizer
        self.pad = self.tokenizer.pad
        self.vocab_size = tokenizer.n_tokens
        
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.embedding = nn.Embedding(
            self.vocab_size, 
            self.emb_size, 
            padding_idx=self.tokenizer.char_to_int[self.pad]
        )
        
        self.gru = nn.GRU(
            self.emb_size,
            self.hidden_size,
            num_layers=self.num_layers,
            #dropout=0.1,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        
        self.latent_mean = nn.Linear(self.hidden_size, self.latent_size)
        self.latent_logvar = nn.Linear(self.hidden_size, self.latent_size)
        
    def forward(self, inputs):
        """
        args:
            - inputs: a batch of SMILES strings [batch_size, seq_len]
                
        returns:
            - mu: mean of Gaussion distribution [batch_size, latent_size]
            - logvar: variation of Gaussion distribution [batch_size, latent_size]
        """
        embed = self.embedding(inputs) # [batch_size, seq_len, emb_size]
        output, hidden = self.gru(embed, None) # output: [batch_size, seq_len, hidden_size*2], hidden: [batch_size, seq_len, hidden_size]
        output = output[:, -1, :].squeeze() # [batch_size, hidden_size*2]
                
        if self.bidirectional:
            output = output[:, :self.hidden_size] + output[:, self.hidden_size:] # [batch_size, hidden_size]
        else:
            output = output[:, :self.hidden_size]
            
        mu = self.latent_mean(output) # [batch_size, latent_size]
        logvar = self.latent_logvar(output) # [batch_size, latent_size]
            
        return mu, logvar

#=============================================
#  Define DecoderRNN: decode Z to SMILES (MolVAE)
class DecoderRNN(nn.Module):
    
    def __init__(
        self, 
        emb_size,
        hidden_size,
        num_layers,
        latent_size,
        condition_size,
        tokenizer
    ):
        
        super(DecoderRNN, self).__init__()
        
        self.tokenizer = tokenizer
        self.start = self.tokenizer.start
        self.vocab_size = tokenizer.n_tokens
        
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = latent_size + condition_size
        
        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        self.gru = nn.GRU(
            self.emb_size + self.input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        
        self.i2h = nn.Linear(self.input_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size + self.input_size, self.vocab_size)
    
    def forward(
        self, 
        inputs,
        z, 
        condition,
        temperature
    ):
        """
        args:
            - inputs: [batch_size, seq_len]
            - z: a batch of latent vectors [batch_size, latent_size]
            - condition: a batch of GX features [batch_size, condition_size]
            - temperature: temperature to smooth the distribution
            
        returns:
            - outputs: output distribution of SMILES strings [batch_size, seq_len, vocab_size]
        """
        model_random_state = np.random.RandomState(1988)
        batch_size, n_steps = inputs.size()
        outputs =torch.zeros(batch_size, n_steps, self.vocab_size).to(get_device())
        input = torch.ones([batch_size, 1], dtype=torch.int32) * self.tokenizer.char_to_int[self.start] # [batch_size, 1]
        input = input.to(get_device())
        decode_embed = torch.cat([z, condition], 1) # [batch_size, latent_size+condition_size]
    
        hidden = self.i2h(decode_embed).unsqueeze(0).repeat(self.num_layers, 1, 1) # [1, batch_size, hidden_size]

        for i in range(n_steps):
            output, hidden = self.step(decode_embed, input, hidden) # output: [batch_size, vocab_size]
            outputs[:, i] = output
            use_teacher_forcing = model_random_state.rand() < temperature
            
            if use_teacher_forcing:
                input = inputs[:, i]
            else:
                input = torch.multinomial(torch.exp(output), 1) # [batch_size, 1]
                
            if input.dim() == 0:
                input = input.unsqueeze(0)
        
        outputs = outputs.squeeze(1) # [batch_size, seq_len, vocab_size]
            
        return outputs
        
    def step(
        self,
        decode_embed,
        input,
        hidden
    ):
        """
        args:
            - decoded_embed: combination of z and condition [batch_size, latent_size+condition_size]
            - input: the stepwise generation [batch_size, 1]
            - hidden: stepwise hidden state of GRU [batch_size, 1, hidden_size]
        
        returns:
            - output: token / atom distribution with [batch_size, vocab_size]
            - hidden: stepwise hidden state of GRU with [1, batch_size, hidden_size]
        """
        input = self.embedding(input).squeeze() # [batch_size, emb_size]
        input = torch.cat((input, decode_embed), 1) # [batch_size, emb_size+latent_size+condition_size]
        input = input.unsqueeze(1)  # [batch_size, 1, emb_size+latent_size+condition_size]
        output, hidden = self.gru(input, hidden) # output: [batch_size, 1, hidden_size], hidden: [1, batch_size, hidden_size]
        output = output.squeeze(1) # [batch_size, hidden_size]
        output = torch.cat((output, decode_embed), 1) # [batch_size, hidden_size+ latent_size+condition_size]
        output = self.out(output) # [batch_size, vocab_size]
        
        return output, hidden

#=============================================
# Define SmilesVAE (MolVAE)
class SmilesVAE(nn.Module):
    
    def __init__(self, encoder, decoder):
        
        super(SmilesVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = nn.CrossEntropyLoss()
        
    def reparameterize(self, mu, logvar):
        
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        
        return mu + eps * std
    
    def forward(
        self,
        inputs,
        condition,
        temperature
    ):
        """
        args: 
            - inputs: a batch of SMILES strings [batch_size, seq_len]
            - condition: a batch of GX features [batch_size, condition_size]
            - temperature: temperature to smooth the distribution
        
        returns:
            - z: a batch of latent vectors [batch_size, latent_size]
            - decoded: a batch of reconstructed SMILES samples [batch_size, seq_len, vocab_size]
        """
        self.mu, self.logvar = self.encoder(inputs)
        z = self.reparameterize(self.mu, self.logvar) # [batch_size, latent_size]
        decoded = self.decoder(inputs, z, condition, temperature) # [batch_size, seq_len, vocab_size]
        
        return z, decoded
    
    def joint_loss(
        self,
        decoded,
        targets,
        alpha=0.5,
        beta=1
    ):
        """
        args:
            - decoded: decoder outputs [batch_size, seq_len, vocab_size]
            - targets: encoder inputs [batch_size, input_size]
            - alpha: L2 loss
            - beta: Scaling of the KLD in range [1, 100]
            
        returns:
            - loss:
        """
        decoded = decoded.permute(0,2,1) # [batch_size, vocab_size, seq_len]
        rec_loss = self.criterion(decoded, targets)
        kld_loss = -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        joint_loss = alpha * rec_loss + (1 - alpha) * beta * kld_loss
        
        return joint_loss, rec_loss, kld_loss
    
    def generation(
        self, 
        rand_z,
        condition,
        max_len,
        tokenizer
    ):
        """
        args:
            - rand_z: sampled latent vectors of SmilesVAE encoder [batch_size, smiles_latent_size]
            - condition: gene expression profile features [batch_size, gene_latent_size] 
            - max_len: maximum length for the generated SMILES strings
            - tokenizer: tokenizer
        returns:
            - generated_smiles_tokens: the generated Smiles tokens [batch_size, max_len]
        """   
        batch_size = rand_z.size(0)
        # Pre-define the output size
        generated_smiles_tokens =torch.zeros(batch_size, max_len).to(get_device())
        # Intput for one time step
        input = torch.ones([batch_size, 1], dtype=torch.int32) * tokenizer.char_to_int[tokenizer.start] # [batch_size, 1]
        input = input.to(get_device())
        # Combine z and condition
        decode_embed = torch.cat([rand_z, condition], 1) # [batch_size, latent_size+condition_size]
        hidden = self.decoder.i2h(decode_embed).unsqueeze(0).repeat(self.decoder.num_layers, 1, 1) # [1, batch_size, hidden_size]

        for i in range(max_len):
            output, hidden = self.decoder.step(decode_embed, input, hidden) # output: [batch_size, vocab_size]
            input = torch.multinomial(torch.exp(output), 1) # [batch_size, 1]
            generated_smiles_tokens[:, i] = input.squeeze(1) # [batch_size, max_len]
            
        return generated_smiles_tokens

    def load_model(self, path):
        weights = torch.load(path)
        self.load_state_dict(weights)

    def save_model(self, path):
        torch.save(self.state_dict(), path)






































    









        
    
    
































