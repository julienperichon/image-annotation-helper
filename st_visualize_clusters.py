from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow_datasets as tfds

STREAMLIT_TITLE = "Clusters selection"
SELECTED_CLUSTERS_FILENAME = "selected_clusters.csv"
CLUSTERS_FILENAME = "clusters.csv"
SAMPLE_SIZE = 100
RANDOM_SEED = 42

CLUSTER_ID_COLUMN = "cluster_id"

dataset = tfds.load("tf_flowers", split="train", as_supervised=True)
images, _ = tuple(zip(*dataset))


def draw_images(df: pd.DataFrame):
    for cluster_id, group_df in df.groupby(CLUSTER_ID_COLUMN):
        st.markdown("---")
        checkbox = st.checkbox(label="Load all images", key=cluster_id)
        if checkbox:
            selected_crops_df = group_df
        else:
            selected_crops_df = group_df.sample(
                n=min(SAMPLE_SIZE, len(group_df)), random_state=RANDOM_SEED
            )
        cluster_title = f"> CLUSTER_[{cluster_id}]"
        with st.expander(cluster_title, expanded=True):
            images_to_draw = get_numpy_images_from_dataset(selected_crops_df)
            st.image(image=images_to_draw)


def get_numpy_images_from_dataset(df: pd.DataFrame):
    numpy_images = []
    df["image_index"].apply(lambda index: numpy_images.append(images[index].numpy()))
    return numpy_images


def main():
    """
    Run Streamlit Application
    """

    st.set_page_config(page_title=STREAMLIT_TITLE, layout="wide", page_icon="🏧")
    st.title(STREAMLIT_TITLE)

    clusters_directory = Path("clusters")
    selected_cluster_path = clusters_directory.parent / SELECTED_CLUSTERS_FILENAME
    load_selected_clusters(selected_cluster_path)

    # Select proper number of clusters for a given macro_label
    st.select_slider(
        "Number of clusters",
        get_cluster_choice_parameter_values(clusters_directory),
        key="selected_parameter",
    )
    load_clusters_for_given_parameter(
        clusters_directory,
        st.session_state.selected_parameter,
    )

    st.button(
        "Save selected clusters",
        help="Click this button to save your results after selecting the best parameter value",
        on_click=save_selected_clusters,
        kwargs={
            "clusters_df": st.session_state.clusters,
            "selected_cluster_path": selected_cluster_path,
        },
    )

    draw_images(st.session_state.clusters)


def get_cluster_choice_parameter_values(clusters_directory: Path) -> np.ndarray:
    """Get values used for cluster choice parameter in compute_clusters step.

    Args:
        clusters_directory (pathlib.Path): Directory in which to look for "{parameter_name}={parameter_value}"
            directories.

    Returns:
        np.ndarray: Values used for cluster choice parameter in compute_clusters step.
    """
    return np.sort(
        np.array(
            [
                float(directory.name.split("=")[1])
                for directory in clusters_directory.glob("n_clusters=*")
            ]
        ).astype(int)
    )


def load_clusters_for_given_parameter(
    clusters_directory: Path, selected_parameter: str
) -> None:
    """
    Load clusters according to selected values on Streamlit app, and save it in st.session_state.clusters

    Args:
        clusters_directory (pathlib.Path): Directory in which to look for "{parameter_name}={parameter_value}"
            directories. Denotes the selected service.
        selected_parameter (str): Selected macro label
    """
    st.session_state.clusters = pd.read_csv(
        clusters_directory / f"n_clusters={selected_parameter}" / CLUSTERS_FILENAME
    ).astype({"image_index": int})


def load_selected_clusters(selected_cluster_path: Path) -> None:
    """Load clusters according to selected values on Streamlit app.

    Args:
        selected_cluster_path (pathlib.Path): Path to the selected_clusters.csv file

    """
    if selected_cluster_path.exists():
        st.session_state.selected_clusters_df = pd.read_csv(
            selected_cluster_path
        ).astype({"image_index": int})
    else:
        st.session_state.selected_clusters_df = pd.DataFrame()


def save_selected_clusters(
    clusters_df: pd.DataFrame, selected_cluster_path: Path
) -> None:
    """Callback used to save selected clusters in a CSV file

    Args:
        clusters_df (pd.DataFrame): Clusters selected. Should at least have these columns :
            [image_name,x1,y1,x2,y2,container,cluster_id].
        selected_cluster_path (pathlib.Path): Path to the directory to save updated clusters
    """

    # Update selected_clusters_df
    st.session_state.selected_clusters_df = clusters_df.astype({CLUSTER_ID_COLUMN: int})

    # Update files
    st.session_state.selected_clusters_df.to_csv(selected_cluster_path, index=False)


if __name__ == "__main__":
    main()
