from pathlib import Path

############################## RAW DATA #################################
DIR = Path("/data1/xp/data/MSBert/GNPS")

RAW = "ALL_GNPS.mgf"
EXTRACTED = "ALL_GNPS.npy"
FILTERED = "filter.npy"
CLEANED = "clean.npy"
POSITIVE = "positive.npy"
ANNOTATED = "annotated.npy"

ORBITRAP = "orbitrap.npy"
QTOF = "qtof.npy"
OTHER = "other.npy"
UNIQUE_SMILES = "unique_smiles.npy"
TANOMOTO_SCORE = "tanimoto_score.npy"

############################# TRAIN_TEST DATA #############################
ALL = "all.npy"
TRAIN_QUERY = "train_query.npy"
TRAIN_REF = "train_ref.npy"
TEST_QUERY = "test_query.npy"
TEST_REF = "test_ref.npy"

ORBITRAP_DIR = DIR / "Orbitrap"
ORBITRAP_DIR.mkdir(exist_ok=True)
ORBITRAP_ALL = ORBITRAP_DIR / ALL
ORBITRAP_TRAIN_QUERY = ORBITRAP_DIR / TRAIN_QUERY
ORBITRAP_TRAIN_REF = ORBITRAP_DIR / TRAIN_REF
ORBITRAP_TEST_QUERY = ORBITRAP_DIR / TEST_QUERY
ORBITRAP_TEST_REF = ORBITRAP_DIR / TEST_REF

QTOF_DIR = DIR / "QTOF"
QTOF_DIR.mkdir(exist_ok=True)
QTOF_ALL = QTOF_DIR / ALL
QTOF_TEST_QUERY = QTOF_DIR / TEST_QUERY
QTOF_TEST_REF = QTOF_DIR / TEST_REF

OTHER_DIR = DIR / "Other"
OTHER_DIR.mkdir(exist_ok=True)
OTHER_ALL = OTHER_DIR / ALL
OTHER_TEST_QUERY = OTHER_DIR / TEST_QUERY
OTHER_TEST_REF = OTHER_DIR / TEST_REF
