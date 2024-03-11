# GxVAEs

A PyTorch implementation of “GxVAEs： Two Joint VAEs Generate Hit Molecules from Gene Expression Profiles“.
The paper has been accepted by [AAAI 2024](https://aaai.org/wp-content/uploads/2024/01/AAAI-24-Oral-Papers-Schedule-3.pdf) (Main track paper and oral presentation). 

![Overview of GxVAEs](https://github.com/naruto7283/GxVAEs/blob/main/overview.png)

## Environment Installation
Execute the following command:
```
$ conda env create -n gxvae_env -f gxvaes_env.yml
$ source activate gxvaes_env
```
Next, download the [dataset](https://drive.google.com/drive/folders/1Bj5CLupoLLCubVx4L7H2yYn62BeAkHYC?usp=sharing) and unzip **datasets.zip** under the path "datasets/LINCS/"

## File Description

- **The datasets Folder**
    - LINCS/mcf7.csv: The training and validation dataset
    - tools floder
- **main.py:**: Defines the main function for training the ProfileVAE and MolVAE models.
- **ProfileVAE.py**: Defines the ProfileVAE model for extracting gene expression profile features.
- **train_gene_vae.py**: Code for training the ProfileVAE model.
- **MolVAE.py**: Defines the MolVAE model for generating SMILES strings with extracted gene features.
- **train_smiles_vae.py**: Code for training the MolVAE model.
- **utils.py**: Defines the functions used.

## Experimental Reproduction

  - **STEP 1**: Pretrain ProfileVAE:
  ``` 
  $ python main.py --train_gene_vae
  ```
  - **STEP 2**: Test the trained ProfileVAE:
  ```
  $ python main.py --test_gene_vae
  ```
  - **STEP 3**: Train MolVAE:
  ```  
  $ python main.py --train_smiles_vae
  ```
  - **STEP 4**: Test the trained MolVAE:
  ```
  $ python main.py --test_smiles_vae
  ```
  - **STEP 5**: Generate molecules for the 10 ligands using GxVAEs
  ```
  $ python main.py --generation
  ```	
  - **STEP 6**: Calculate Tanimoto similarity between a source ligand and generated SMILES strings: 
  ```
  $ python main.py --calculate_tanimoto --protein_name XXX***
  ```
  
## Citation
  ```
  C. Li and Y. Yamanishi (2024). GxVAEs： Two Joint VAEs Generate Hit Molecules from Gene Expression Profiles.
  ```
  
  BibTeX format:
  ```
  @inproceedings{li2024gxvaes,
  title={GxVAEs： Two Joint VAEs Generate Hit Molecules from Gene Expression Profiles},
  author={Li, Chen and Yamanishi, Yoshihiro},
  booktitle={Proceedings of the 38th AAAI Conference on Artificial Intelligence (AAAI 2024)},
  year={2024}
}
  ```
