import ast
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow_datasets as tfds

from st_visualize_clusters import CLUSTER_ID_COLUMN

STATUS_OK = "OK"
STATUS_KO = "KO"
STATUS_TODO = "TODO"

CLUSTER_NAME_COLUMN = "cluster_name"

DELETE_CLUSTER_NAME = "DELETE"

LABELLED_CLUSTERS_FILENAME = Path("labelled_clusters.csv")
SELECTED_CLUSTERS_FILENAME = Path("selected_clusters.csv")


def display_and_label_crops(selected_cluster_id: str) -> None:

    #%% Select crops for given selected macro label and initialize status to OK
    crops_per_cluster_id = st.session_state.clusters.loc[
        lambda df: df[CLUSTER_ID_COLUMN] == selected_cluster_id
    ].replace({"status": STATUS_TODO}, STATUS_OK)

    #%% Draw crops with clickable_images_gallery component
    with st.form("Draw crops"):
        st.text_input(
            "Cluster label",
            key=f"text_input_{selected_cluster_id}",
        )
        st.checkbox("Delete cluster ?", key=f"delete_checkbox_{selected_cluster_id}")

        image_columns = st.columns(st.session_state.num_images_per_line)
        column_index = 0

        st.session_state.images_indexes = crops_per_cluster_id.image_index.unique()
        for image_index in st.session_state.images_indexes:
            with image_columns[column_index]:
                st.image(
                    st.session_state.images[image_index].numpy(),
                    caption=f"{image_index}",
                )
                st.checkbox(
                    "Drop outlier?", key=f"checkbox_{selected_cluster_id}_{image_index}"
                )
                st.markdown("""---""")
                column_index = (column_index + 1) % st.session_state.num_images_per_line

        st.form_submit_button(
            "Save (overwrite) catalog",
            on_click=update_catalog_file,
            kwargs={
                "crops_per_cluster_id": crops_per_cluster_id,
            },
        )


def update_catalog_file(crops_per_cluster_id: pd.DataFrame) -> None:
    labelled_cluster_path = Path(LABELLED_CLUSTERS_FILENAME)
    st.session_state.clusters = pd.concat(
        [
            st.session_state.clusters.loc[
                lambda df: df[CLUSTER_ID_COLUMN] != st.session_state.selected_cluster_id
            ],
            get_updated_image_data(
                crops_with_old_data=crops_per_cluster_id[
                    st.session_state.clusters.columns
                ]
            ),
        ]
    ).sort_index()

    st.session_state.clusters.to_csv(labelled_cluster_path, index=False)

    st.session_state.selected_cluster_id = get_next_cluster_id()


def clear_session_state() -> None:
    for key in st.session_state.keys():
        if key != "restaurant_name":
            del st.session_state[key]


def get_updated_image_data(crops_with_old_data: pd.DataFrame):
    crops_with_updated_status = get_clickable_images_gallery_updated(
        crops_with_old_status=crops_with_old_data,
    )
    crops_with_updated_cluster_names = get_cluster_names_updated(
        crops_with_old_cluster_labels=crops_with_updated_status,
        cluster_label_text_input_keys_dict={
            cluster_id: f"text_input_{cluster_id}"
            for cluster_id in crops_with_updated_status[CLUSTER_ID_COLUMN].unique()
        },
        delete_checkbox_keys_dict={
            cluster_id: f"delete_checkbox_{cluster_id}"
            for cluster_id in crops_with_updated_status[CLUSTER_ID_COLUMN].unique()
        },
    )
    return crops_with_updated_cluster_names


def get_clickable_images_gallery_updated(
    crops_with_old_status: pd.DataFrame,
) -> pd.DataFrame:
    crops_with_updated_status = crops_with_old_status.copy()
    for image_index in st.session_state.images_indexes:
        status_ko = st.session_state[
            f"checkbox_{st.session_state.selected_cluster_id}_{image_index}"
        ]
        crops_with_updated_status.loc[
            crops_with_updated_status.image_index == image_index, "status"
        ] = (STATUS_KO if status_ko else STATUS_OK)
    return crops_with_updated_status

    # objects_with_status = crops_with_old_status.assign(
    #     id=lambda df: pd.util.hash_pandas_object(df[BoxColumnsKey], index=False).astype(
    #         str
    #     )
    # )
    # status_mapping = (
    #     objects_with_status[["id", "status"]].set_index("id").status.to_dict()
    # )
    # for component_key in clickable_images_gallery_keys:
    #     component_values = st.session_state[component_key]
    #     if component_values is not None:
    #         status_mapping.update(
    #             {
    #                 crop["id"]: crop["status"]
    #                 for crop in ast.literal_eval(component_values)
    #             }
    #         )
    # return objects_with_status.assign(status=lambda df: df.id.map(status_mapping)).drop(
    #     columns="id"
    # )


def get_cluster_names_updated(
    crops_with_old_cluster_labels: pd.DataFrame,
    cluster_label_text_input_keys_dict: Dict[str, str],
    delete_checkbox_keys_dict: Dict[str, str],
) -> pd.DataFrame:
    for cluster_id, text_input_key in cluster_label_text_input_keys_dict.items():
        crops_with_old_cluster_labels.loc[
            crops_with_old_cluster_labels[CLUSTER_ID_COLUMN] == cluster_id,
            CLUSTER_NAME_COLUMN,
        ] = st.session_state[text_input_key].upper()

    for cluster_id, delete_checkbox_key in delete_checkbox_keys_dict.items():
        if st.session_state[delete_checkbox_key]:
            crops_with_old_cluster_labels.loc[
                crops_with_old_cluster_labels[CLUSTER_ID_COLUMN] == cluster_id,
                CLUSTER_NAME_COLUMN,
            ] = DELETE_CLUSTER_NAME

    return crops_with_old_cluster_labels


def load_clusters() -> None:
    if "clusters" not in st.session_state:
        if not LABELLED_CLUSTERS_FILENAME.exists():
            annotations_df = (
                pd.read_csv(SELECTED_CLUSTERS_FILENAME)
                .assign(status=STATUS_TODO, cluster_name="")
                .groupby(CLUSTER_ID_COLUMN, as_index=False)
                .apply(lambda g: g.sample(len(g)))
            )
            annotations_df.to_csv(
                LABELLED_CLUSTERS_FILENAME,
                index=False,
            )
        else:
            annotations_df = pd.read_csv(LABELLED_CLUSTERS_FILENAME)
        st.session_state.clusters = annotations_df


def get_next_cluster_id() -> str:
    if ~(st.session_state.clusters.status == STATUS_TODO).any():
        return (
            st.session_state.clusters[CLUSTER_ID_COLUMN]
            .drop_duplicates()
            .sort_values()
            .iloc[-1]
        )

    return (
        st.session_state.clusters.loc[lambda df: df.status == STATUS_TODO][
            CLUSTER_ID_COLUMN
        ]
        .drop_duplicates()
        .sort_values()
        .iloc[0]
    )


def get_task_progression(macro_labels: np.ndarray, macro_label_todo: str) -> float:
    return int(np.where(macro_labels == macro_label_todo)[0][0]) / len(macro_labels)


# %% Streamlit App
def main() -> None:
    """
    Run Streamlit Application
    """
    st.set_page_config(
        page_title="Label clusters and outliers", layout="wide", page_icon="📚"
    )
    st.title("Label clusters and outliers")

    dataset = tfds.load("tf_flowers", split="train", as_supervised=True)
    st.session_state.images, _ = tuple(zip(*dataset))

    #%% Read catalog
    load_clusters()

    #%% Select macro_labels
    cluster_ids = (
        st.session_state.clusters[CLUSTER_ID_COLUMN]
        .drop_duplicates()
        .sort_values()
        .to_list()
    )
    cluster_id_todo = get_next_cluster_id()
    if "selected_cluster_id" not in st.session_state:
        st.session_state.selected_cluster_id = cluster_id_todo
    selected_cluster_id = st.sidebar.selectbox(
        "Cluster", options=cluster_ids, key="selected_cluster_id"
    )

    st.sidebar.number_input(
        "Number of images per line",
        min_value=1,
        value=5,
        step=1,
        key="num_images_per_line",
    )

    #%% Display and annotate
    st.progress(get_task_progression(cluster_ids, cluster_id_todo))
    display_and_label_crops(selected_cluster_id)


if __name__ == "__main__":
    main()
