from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import typer
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

CLUSTERS_DIR = Path("clusters")


def compute_clusters(
    n_clusters_min: int = typer.Option(...), n_clusters_max: int = typer.Option(...)
) -> None:
    # Load model
    model = tf.keras.Sequential(
        [
            hub.KerasLayer(
                "https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/5",
                trainable=False,
            ),
        ]
    )
    model.build([None, 299, 299, 3])  # Batch input shape.

    # Load dataset
    dataset = tfds.load("tf_flowers", split="train", as_supervised=True)

    # Preprocess data
    dataset = preprocess_dataset(dataset)

    # Compute embeddings
    typer.echo("Compute image embeddings from model...")
    predictions_per_batch = []
    for batch_images, _ in tqdm(dataset):
        predictions_per_batch.append(model.predict(batch_images))
    embeddings = np.concatenate(predictions_per_batch)
    typer.echo("Image embeddings computation finished !\n")

    # Cluster images
    typer.echo("Cluster images with embeddings...")
    for n_clusters in range(n_clusters_min, n_clusters_max + 1):
        typer.echo(f"Compute for n_clusters={n_clusters}...")
        agglomerative_clustering_instance = AgglomerativeClustering(
            n_clusters=n_clusters, linkage="ward"
        )
        cluster_ids = agglomerative_clustering_instance.fit_predict(embeddings)
        results_dir = CLUSTERS_DIR / f"n_clusters={n_clusters}"
        results_dir.mkdir(exist_ok=True, parents=True)
        (
            pd.DataFrame(data={"cluster_id": cluster_ids})
            .assign(
                image_index=lambda df: df.index,
            )
            .to_csv(results_dir / "clusters.csv", index=False)
        )


def preprocess_dataset(dataset):
    resizing_layer = tf.keras.layers.Resizing(299, 299)
    return (
        dataset.cache()
        .map(lambda x, y: (resizing_layer(x), y))
        .map(lambda x, y: (x / 255.0, y))
        .batch(100)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )


if __name__ == "__main__":
    typer.run(compute_clusters)
