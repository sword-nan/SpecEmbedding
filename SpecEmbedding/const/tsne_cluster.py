from pathlib import Path

MSBERT_DIR = Path("/data1/xp/data/MSBert/GNPS")

MSBERT_TSNE_RAW = MSBERT_DIR / "tsne.mgf"
MSBERT_CLUSTER_RAW = MSBERT_DIR / "cluster.mgf"

MSBERT_TSNE = MSBERT_DIR / "tsne.npy"
MSBERT_CLUSTER = MSBERT_DIR / "cluster.npy"

SPECEMBEDDING_DIR = Path("/data1/xp/code/specEmbedding/tsne_cluster_data")

SPECEMBEDDING_TSNE = SPECEMBEDDING_DIR / "tsne.npy"
SPECEMBEDDING_TSNE_RAW = SPECEMBEDDING_DIR / "tsne.mgf"
SPECEMBEDDING_CLUSTER = SPECEMBEDDING_DIR / "cluster.npy"
SPECEMBEDDING_CLUSTER_RAW = SPECEMBEDDING_DIR / "cluster.mgf"