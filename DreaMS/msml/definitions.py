import sys
import inspect
import pathlib

# Root path
MSML_DIR = pathlib.Path(__file__).parent.absolute()

# Paths to datasets and their dirs
DATA_DIR = MSML_DIR / 'data'
NIST20 = DATA_DIR / 'NIST20' / 'datasets'
NIST20_PKL = NIST20 / 'nist20_clean.pkl'
MONA = DATA_DIR / 'MoNA' / 'datasets'
MONA_PKL = MONA / 'mona_clean.pkl'
CASMI = DATA_DIR / 'CASMI' / 'datasets'
CASMI_PKL = CASMI / 'casmi16.pkl'
MASSIVE_DIR = DATA_DIR / 'MassIVE'
MASSIVE_DATA = MASSIVE_DIR / 'data'
MASSIVE_SCRIPTS = MASSIVE_DIR / 'scripts'
MCE = DATA_DIR / 'MCE' / 'datasets'
MIST = DATA_DIR / 'MIST' / 'datasets'
MERGED_DATASETS = DATA_DIR / 'merged' / 'datasets'

# Molecules
PUBCHEM = DATA_DIR / 'molecules' / 'pubchem'
PUBCHEM_FPS = PUBCHEM / 'cid_smiles_index.h5'

# Experiments
EXPERIMENTS_DIR = MSML_DIR / 'experiments'

# Misc
MISC = MSML_DIR / 'misc'

# Figures
FIGURES = MISC / 'figures'

# Pre-trained models
PRETRAINED = MSML_DIR / 'models' / 'checkpoints'

# Data column names
SPECTRUM = 'spectrum'
PRECURSOR_MZ = 'precursor_mz'
CHARGE = 'charge'
ADDUCT = 'adduct'
SMILES = 'smiles'
MSDATA_COLUMNS = [SPECTRUM, PRECURSOR_MZ, CHARGE, ADDUCT]
MSDATA_LABELED_COLUMNS = MSDATA_COLUMNS + [SMILES]

DATASET = 'dataset'
ID = 'id'
DREAMS_EMBEDDING = 'DreaMS_embedding'
RT = 'RT'

def export():
    # Export paths
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if isinstance(obj, pathlib.PosixPath):
            path = str(obj)
            print(f'export {name}={path}')
