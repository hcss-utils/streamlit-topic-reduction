# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import streamlit as st
import datashader as ds
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

st.set_page_config(layout="wide")
st.set_option("deprecation.showPyplotGlobalUse", False)
ROOT = Path(__file__).resolve().parent.parent
XRANGE = (0, 20)
YRANGE = (-7.5, 12)


def _to_hex(arr):
    return [matplotlib.colors.to_hex(c) for c in arr]


def _get_extent(points):
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])

    extent = (
        np.round(min_x - 0.05 * (max_x - min_x)),
        np.round(max_x + 0.05 * (max_x - min_x)),
        np.round(min_y - 0.05 * (max_y - min_y)),
        np.round(max_y + 0.05 * (max_y - min_y)),
    )
    return extent


def _select_dist(data, num):
    df = data.rename(
        columns={
            f"num_label_{num}": "reduced_topics",
            f"word_label_{num}": "reduced_words",
        }
    ).copy()
    df = df.loc[:, ["x", "y", "reduced_topics", "reduced_words"]]
    df["reduced_topics"] = pd.Categorical(df["reduced_topics"])
    df["reduced_words"] = pd.Categorical(df["reduced_words"])
    return df


def points(
    data,
    num_topics,
    plotting_method,
    zooming,
    labels_type,
    width=800,
    height=800,
):
    df = _select_dist(data, num_topics)
    point_size = 100.0 / np.sqrt(df.shape[0])

    dpi = plt.rcParams["figure.dpi"]
    fig = plt.figure(figsize=(width / dpi, height / dpi))
    ax = fig.add_subplot(111)

    _label = "reduced_topics" if labels_type == "topic number" else "reduced_words"
    unique_labels = df[_label].unique()
    num_labels = unique_labels.shape[0]
    color_key = _to_hex(plt.get_cmap("Spectral")(np.linspace(0, 1, num_labels)))
    legend_elements = [
        Patch(facecolor=color_key[i], label=k + 1 if isinstance(k, int) else k)
        for i, k in enumerate(sorted(unique_labels))
    ]

    if plotting_method == "matplotlib":
        new_color_key = {
            k: matplotlib.colors.to_hex(color_key[i])
            for i, k in enumerate(sorted(unique_labels))
        }
        colors = df[_label].map(new_color_key)
        ax.scatter(df["x"], df["y"], s=point_size, c=colors)
        if zooming:
            plt.xlim(XRANGE)
            plt.ylim(YRANGE)
    else:
        extent = _get_extent(df[["x", "y"]].values)
        cvs = ds.Canvas(
            plot_width=width,
            plot_height=height,
            x_range=XRANGE if zooming else (extent[0], extent[1]),
            y_range=YRANGE if zooming else (extent[2], extent[3]),
        )
        agg = cvs.points(df, "x", "y", agg=ds.count_cat(_label))
        result = ds.tf.shade(agg, color_key=color_key, how="eq_hist")
        img_rgba = result.data.view(np.uint8).reshape(result.shape + (4,))
        ax.imshow(img_rgba[::-1], extent=extent)

    ax.set_title(f"Reduced 2D embeddings: {num_topics} labeled topics")
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set(xticks=[], yticks=[])


@st.cache
def load_model(dataset):
    filename = (
        "fastlearn.csv"
        if dataset == "noun phrases"
        else "deciding-on-topic-reduction.csv"
    )
    return pd.read_csv(ROOT / "data" / filename)


st.title("Hierarchical topic reduction")
st.write(
    """
We used Top2Vec algorithm to automatically detect topics 
present in our ProQuest Deterrence dataset. 

The model initially found 229 topics on raw text and 49 topics on noun phrases - both of which is too much to interpret. Thus, 
we decided to reduce the number of topics using top2vec's hierarchical topic reduction method.
"""
)

st.sidebar.markdown("# Controls")
st.sidebar.markdown("You can **change** the values to change the *chart*.")
st.sidebar.markdown("We trained a model using both *raw text* and *noun phrases*.")
dataset = st.sidebar.selectbox("Select source data", ("raw text", "noun phrases"))
num_topic = st.sidebar.slider("Number of topics", min_value=2, max_value=20, step=1)
st.sidebar.markdown(
    """
You can also use different approaches to plotting:
* matplotlib
* datashader (for large datasets)
"""
)
plotting_method = st.sidebar.selectbox(
    "Select plotting method", ("matplotlib", "datashader")
)

st.sidebar.markdown("To *zoom in*, fill checkbox below")
zoom = st.sidebar.checkbox("Zoomed in")

labels_type = st.sidebar.selectbox(
    "Select legend labels", ("topic number", "most common words")
)

df = load_model(dataset)
plot = points(df, num_topic, plotting_method, zoom, labels_type)
st.pyplot(plot)
