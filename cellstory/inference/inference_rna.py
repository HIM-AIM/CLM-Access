import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)


def append_to_obsm(adata, obsm_key, embedding):
    adata.obsm[obsm_key] = embedding


def plot_umap_raw(
    adata,
    umap_png,
    ax,
    umap_title=None,
    n_neighbors=30,
    key="X",
    layer_key="counts",
    celltype_key="cell_type",
    umap_key="X_umap",
    seed=42,
):
    neighbors_key = f"{key}_neighbors"
    leiden_key = f"{key}_leiden"

    # use copied raw counts layer
    adata.X = adata.layers[layer_key].copy()
    # Normalizing to median total counts
    sc.pp.normalize_total(adata)
    # Logarithmize the data
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    sc.tl.pca(adata)
    sc.pp.neighbors(
        adata, n_neighbors=30, n_pcs=50, use_rep=None, key_added=neighbors_key
    )
    sc.tl.leiden(adata, key_added=leiden_key, neighbors_key=neighbors_key)
    sc.tl.umap(adata, neighbors_key=neighbors_key)
    # plot single umap & return
    umap_fig = sc.pl.umap(
        adata,
        color=[celltype_key],
        title=umap_title,
        neighbors_key=neighbors_key,
        return_fig=True,
        show=False,
    )
    # save umap
    umap_fig.savefig(umap_png, bbox_inches="tight")
    # plot on ax
    sc.pl.umap(
        adata,
        color=[celltype_key],
        title=umap_title,
        neighbors_key=neighbors_key,
        return_fig=False,
        show=False,
        ax=ax,
    )
    metrics = kmeans_umap(
        adata, umap_key=umap_key, celltype_key=celltype_key, seed=seed
    )
    # return values
    return umap_fig, metrics


def plot_umap_embed(
    adata,
    umap_png,
    ax,
    umap_title=None,
    n_neighbors=30,
    key="cellstory_rna",
    celltype_key="cell_type",
    umap_key="X_umap",
    seed=42,
):
    rep_key = key
    neighbors_key = f"{key}_neighbors"
    leiden_key = f"{key}_leiden"
    sc.pp.neighbors(adata, n_neighbors=30, use_rep=rep_key, key_added=neighbors_key)
    sc.tl.leiden(adata, key_added=leiden_key, neighbors_key=neighbors_key)
    sc.tl.umap(adata, neighbors_key=neighbors_key)
    # plot single umap & return
    umap_fig = sc.pl.umap(
        adata,
        color=[celltype_key],
        title=umap_title,
        neighbors_key=neighbors_key,
        return_fig=True,
        show=False,
    )
    # save umap
    umap_fig.savefig(umap_png, bbox_inches="tight")
    # plot on ax
    sc.pl.umap(
        adata,
        color=[celltype_key],
        title=umap_title,
        neighbors_key=neighbors_key,
        return_fig=False,
        show=False,
        ax=ax,
    )
    metrics = kmeans_umap(
        adata, umap_key=umap_key, celltype_key=celltype_key, seed=seed
    )
    # return values
    return umap_fig, metrics


def kmeans_umap(adata, umap_key="X_umap", celltype_key="cell_type", seed=42):
    n_clusters = adata.obs[celltype_key].nunique()
    # 使用 K-means 聚类对 UMAP 结果进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)  # 假设有10个聚类
    labels_pred = kmeans.fit_predict(adata.obsm[umap_key])

    # 计算和打印聚类指标
    labels_true = adata.obs[celltype_key]
    # silhouette = silhouette_score(adata.obsm[umap_key], labels_pred, metric="euclidean")
    ari = adjusted_rand_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)

    metrics = {"ARI": ari, "NMI": nmi}
    return metrics


def generate_rna_metrics(args, inferred_adata):
    rna_h5ad_stem = args.rna_h5ad.stem
    embed_h5ad = args.dirpath / f"{rna_h5ad_stem}.{args.obsm_key}.h5ad"
    compare_umap_png = args.dirpath / f"{rna_h5ad_stem}.umap.compare.png"
    raw_umap_png = args.dirpath / f"{rna_h5ad_stem}.umap.raw.png"
    embed_umap_png = args.dirpath / f"{rna_h5ad_stem}.umap.{args.obsm_key}.png"
    umap_metric_tsv = args.dirpath / f"{rna_h5ad_stem}.umap.metrics.tsv"

    fig, ((embed_ax, raw_ax)) = plt.subplots(
        2, 1, figsize=(6.4, 9.6), gridspec_kw=dict(wspace=0.5)
    )
    # embedding umap
    umap_fig_embed, embed_metrics = plot_umap_embed(
        inferred_adata,
        umap_png=str(embed_umap_png),
        ax=embed_ax,
        umap_title=args.obsm_key,
        n_neighbors=30,
        key="cellstory_rna",
        celltype_key="cell_type",
        umap_key="X_umap",
        seed=args.seed,
    )
    # raw counts umap
    umap_fig_raw, raw_metrics = plot_umap_raw(
        inferred_adata,
        umap_png=str(raw_umap_png),
        ax=raw_ax,
        umap_title="raw_counts",
        n_neighbors=30,
        key="X",
        layer_key="counts",
        celltype_key="cell_type",
        umap_key="X_umap",
        seed=args.seed,
    )
    # save compare figure
    fig.suptitle(f"{args.obsm_key} vs raw_counts")
    fig.savefig(str(compare_umap_png), bbox_inches="tight")
    # save metric df
    metric_df = pd.DataFrame(
        [raw_metrics, embed_metrics], index=["raw_counts", f"{args.obsm_key}"]
    )
    metric_df.to_csv(umap_metric_tsv, sep="\t")
    # write extracted h5ad
    inferred_adata.write_h5ad(str(embed_h5ad))
    return fig, metric_df
