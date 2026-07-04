import torch
import numpy as np
import pandas as pd
import warnings

from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset

from biotite.structure.io.pdb import PDBFile
from biotite.structure import filter_amino_acids, get_residue_starts

from FM_HiREF.protein import constants as rc

from tqdm import tqdm




class PDBDataset(Dataset):
    def __init__(self, pdb_dir, scale=1.):
        self.pdb_dir = Path(pdb_dir)
        self.pdb_files = sorted(self.pdb_dir.glob('*.pdb'))
        self.cache_file = self.pdb_dir / "pdb_metadata.csv"
        self.scale = scale
        print(f'Number of PDB Files: {len(self.pdb_files)}')

        # Check if cache already exists
        cache_lookup = {}
        if self.cache_file.exists():
            df = pd.read_csv(self.cache_file)
            cache_lookup = dict(zip(df['stem'], df['length']))
        else:
            with open(self.cache_file, 'w') as f:
                f.write("stem,length\n")

        # Load in data from metadata cache, if data not in metadata cache retrieve it
        self.lengths = []
        for file in self.pdb_files:
            if file.stem in cache_lookup:
                self.lengths.append(cache_lookup[file.stem])
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pdb = PDBFile.read(file)
                    atoms = pdb.get_structure(model=1)
        
                n_res = (filter_amino_acids(atoms) & (atoms.atom_name == "CA")).sum()
                cache_lookup[file.stem] = n_res
                self.lengths.append(n_res)
                
                with open(self.cache_file, 'a') as f:
                    f.write(f"{file.stem},{n_res}\n")
        
        # Cluster indices by length - For splitting data into train, val and test
        self.length_clusters = defaultdict(list)
        for i, length in enumerate(self.lengths):
            self.length_clusters[length].append(i)
 
        # Set up pt cache directory
        self.pt_cache_dir = self.pdb_dir / f"pt_cache_scale_{self.scale}"
        self.pt_cache_dir.mkdir(exist_ok=True)

    @classmethod
    def cache_pdbs(cls, pdb_dir, scale=1.0):
        pdb_dir = Path(pdb_dir)
        pt_cache_dir = pdb_dir / f"pt_cache_scale_{scale}"
        pt_cache_dir.mkdir(exist_ok=True)
        pdb_files = list(pdb_dir.glob('*.pdb'))


        print(f"Pre-caching {len(pdb_files)} files with scale {scale}...")
        
        for file in tqdm(pdb_files, desc="Caching Data"):
            cache_file = pt_cache_dir / f"{file.stem}.pt"
            
            # If the file exists in this scale's specific folder, we trust it.
            if not cache_file.exists():
                feats = cls.process_monomer(file, scale)
                torch.save(feats, cache_file)
                
        print("Dataset caching complete")

    @staticmethod
    def process_monomer(pdb_file, scale=1.):
        # Biotite spews out aa lot of user warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pdb = PDBFile.read(pdb_file)
            atom_arr = pdb.get_structure(model=1)


        atom_arr = atom_arr[filter_amino_acids(atom_arr)]
        ca_atoms = atom_arr[atom_arr.atom_name == 'CA']
        n_res = len(ca_atoms)
        
        # Atom 37 feats
        res_bounds = get_residue_starts(atom_arr, add_exclusive_stop=True)
        atom37 = torch.zeros((n_res, 37, 3), dtype=torch.float32)
        atom37_mask = torch.zeros((n_res, 37), dtype=torch.float32)
        for i in range(n_res):
            start = res_bounds[i]
            end = res_bounds[i+1]
            residue_atoms = atom_arr[start:end]
            for atom in residue_atoms:
                if atom.atom_name in rc.ATOM_ORDER:
                    atom_idx = rc.ATOM_ORDER[atom.atom_name]
                    atom37[i, atom_idx] = torch.tensor(atom.coord)
                    atom37_mask[i, atom_idx] = 1.
        
        # Centering atom37 and obtaining backbone feats
        ca_idx = rc.ATOM_ORDER['CA']
        atom_ca_mask = atom37_mask[:, ca_idx]
        bb_center = torch.sum(atom37[:, ca_idx], dim=0) / (torch.sum(atom_ca_mask) + 1e-5)
        atom37 = atom37 - bb_center[None, None, :]   # Centered
        atom37 = atom37 / scale                      # Scaled
        atom37 = atom37 * atom37_mask[..., None]

        atom_ca = atom37[..., ca_idx, :]

        # Aatype feat
        restype_3 = ca_atoms.res_name
        aatype =  [rc.RESTYPE_3TO1.get(res3, 'UNK') for res3 in restype_3]
        aatype = torch.tensor([rc.RESTYPE_ORDER.get(res, 20) for res in aatype], dtype=torch.int64)

        # seq_id and res_id
        res_id = torch.tensor(ca_atoms.res_id)
        seq_pos = torch.arange(len(aatype))

        return {'aatype'        : aatype,
                'seq_pos'       : seq_pos,
                'res_id'        : res_id,
                'atom37'        : atom37,
                'atom37_mask'   : atom37_mask,
                'atom_CA'       : atom_ca,
                'atom_CA_mask'  : atom_ca_mask,
                'scale'         : scale}

    def __len__(self):
        return len(self.pdb_files)

    def __getitem__(self, idx):
        file = self.pdb_files[idx]

        cache_file = self.pt_cache_dir / f"{file.stem}.pt"
        feats = torch.load(cache_file, map_location="cpu", weights_only=True)
        
        # Later used for generating globally aligned noise
        feats['label'] = idx
        
        return feats