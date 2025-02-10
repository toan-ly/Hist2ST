import os
import scanpy as sc
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")



def preprocess_dlpfc():
    """
    Preprocess DLPFC data to identify highly variable genes across all batches
    and save them for use in the ViT_DLPFC dataset class.
    """
    base_dir = 'data/st_data/DLPFC_new'
    # Use actual folder names
    batches = [
        '151507', '151508', '151509', '151510',
        '151669', '151670', '151671', '151672',
        '151673', '151674', '151675', '151676'
    ]
    
    # List to store AnnData objects
    adatas = []
    
    # Load and preprocess each batch
    for batch in batches:
        print(f"Processing {batch}...")
        batch_path = f"{base_dir}/{batch}"
        
        # Load data
        adata = sc.read_visium(batch_path, count_file='filtered_feature_bc_matrix.h5')   
        adata.var_names_make_unique()
        
        metadata_path = os.path.join(batch_path, 'metadata.tsv')
        positions_path = os.path.join(batch_path, 'spatial', 'tissue_positions_list.csv')

        metadata = pd.read_csv(metadata_path, sep='\t')
        spatial_positions = pd.read_csv(positions_path, header=None)
        
        adata.obs = metadata
        
        # Add batch info
        adata.obs['batch'] = batch

        # Store raw counts in layers
        adata.layers['count'] = adata.X.copy()
        
        # Basic filtering
        sc.pp.filter_genes(adata, min_cells=50)
        sc.pp.filter_genes(adata, min_counts=10)
        
        # Normalize
        sc.pp.normalize_total(adata, target_sum=1e6)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=1000)
        
        adatas.append(adata)
    
    # Concatenate all batches
    print("Concatenating batches...")
    combined = adatas[0].concatenate(adatas[1:])
    
    # Calculate highly variable genes
    print("Calculating highly variable genes...")
    sc.pp.highly_variable_genes(
        combined,
        flavor="seurat_v3",
        layer='count',
        n_top_genes=1000,
        batch_key='batch'  # Account for batch effects
    )
    
    # Get the highly variable genes
    hvg = combined.var.index[combined.var.highly_variable].values
        
    # Save HVG list
    print("Saving highly variable genes...")
    np.save('data/dlpfc_hvg_cut_1000.npy', hvg)
    
    print(f"Saved {len(hvg)} highly variable genes")
    
    return hvg

if __name__ == "__main__":
    hvg = preprocess_dlpfc()