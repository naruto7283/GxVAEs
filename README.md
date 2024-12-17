# GxVAEs
A PyTorch implementation of “GxVAEs: Two Joint VAEs Generate Hit Molecules from Gene Expression Profiles“.
The paper has been accepted by [AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/29248). 

![Overview of GxVAEs](https://github.com/naruto7283/GxVAEs/blob/main/overview.png)

## Objectives of GxVAEs
This implementation was developed by Chen Li (li.chen.z2@a.mail.nagoya-u.ac.jp) and Yoshihiro Yamanishi (yamanishi@i.nagoya-u.ac.jp), affiliated with the Department of Complex Systems Science at the Graduate School of Informatics, Nagoya University, Japan, at the time of release.

GxVAEs aim to
- generate hit-like molecules from gene expression profiles.
- generate therapeutic molecules from patients’ disease profiles.

## Environment Installation
Execute the following command:
```
$ conda env create -n gxvae_env -f gxvaes_env.yml
$ source activate gxvaes_env
```

## File Description

- **The datasets Folder**
    - LINCS/mcf7.csv: The training and validation datasets, which consist of gene expression profiles of the MCF7 cell line treated with 13,755 molecules, were used.
    - tools floder
- **main.py:**: Define the main function for training the ProfileVAE and MolVAE models.
- **ProfileVAE.py**: Defines the ProfileVAE model for extracting gene expression profile features.
- **train_gene_vae.py**: Code for training the ProfileVAE model.
- **MolVAE.py**: Defines the MolVAE model to generate SMILES strings with extracted gene features.
- **train_smiles_vae.py**: Code for training the MolVAE model.
- **utils.py**: Defines other functions used in GxVAEs.

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
  $ python main.py --calculate_tanimoto --protein_name ***
  ```
&nbsp;&nbsp;&nbsp;&nbsp;Note that '***' indicates a protein name, such as 'AKT1'.

## Contact
If you have any questions, please feel free to contact Chen Li at li.chen.z2@a.mail.nagoya-u.ac.jp.
  
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
