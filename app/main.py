# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import streamlit as st
import datashader as ds
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

st.set_option('deprecation.showPyplotGlobalUse', False)
root = Path(__file__).resolve().parent


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
    df = data.rename(columns={f"doc_top_reduced_{num}": "reduced_topics"}).copy()
    df = df.loc[:, ~df.columns.str.contains("doc_top_reduced_")]
    df["reduced_topics"] = pd.Categorical(df["reduced_topics"])
    return df

def matplotlib_points(
    data,
    num_topics,
    width=800, 
    height=800,
):
    df = _select_dist(data, num_topics)
    point_size = 100.0 / np.sqrt(df.shape[0])
    
    dpi = plt.rcParams["figure.dpi"] 
    fig = plt.figure(figsize=(width / dpi, height / dpi))
    ax = fig.add_subplot(111) 
    
    unique_labels = df["reduced_topics"].unique()
    num_labels = unique_labels.shape[0]
    color_key = _to_hex(
        plt.get_cmap("Spectral")(np.linspace(0, 1, num_labels))
    )
    new_color_key = {
        k: matplotlib.colors.to_hex(color_key[i])
        for i, k in enumerate(sorted(unique_labels))
    }
    legend_elements = [
        Patch(facecolor=color_key[i], label=k)
        for i, k in enumerate(sorted(unique_labels))
    ]
    
    colors = df["reduced_topics"].map(new_color_key)
    
    ax.scatter(df["x"], df["y"], s=point_size, c=colors)
    ax.set_title(f"Reduced 2D embeddings: {num_topics} labeled topics")
    ax.legend(handles=legend_elements)
    # ax.margins(x=0, y=-.2)
    ax.set(xticks=[], yticks=[])


def datashader_points(
    data,
    num_topics,
    width=800, 
    height=800,
):
    dpi = plt.rcParams["figure.dpi"] 
    fig = plt.figure(figsize=(800 / dpi, 800 / dpi))
    ax = fig.add_subplot(111) 
    
    df = _select_dist(data, num_topics)
    extent = _get_extent(df[["x", "y"]].values)
    
    cvs = ds.Canvas(
        plot_width=width,
        plot_height=height,
        x_range=(extent[0], extent[1]),
        y_range=(extent[2], extent[3]),
    )
    agg = cvs.points(df, "x", "y", agg=ds.count_cat("reduced_topics"))

    unique_labels = df["reduced_topics"].unique()
    num_labels = unique_labels.shape[0]
    color_key = _to_hex(
        plt.get_cmap("Spectral")(np.linspace(0, 1, num_labels))
    )
    legend_elements = [
        Patch(facecolor=color_key[i], label=k)
        for i, k in enumerate(sorted(unique_labels))
    ]
    result = ds.tf.shade(agg, color_key=color_key, how="eq_hist")
    
    #
    ax.imshow(result.data.view(np.uint8).reshape(result.shape + (4,)))
    ax.set_title(f"Reduced 2D embeddings: {num_topics} labeled topics")
    ax.legend(handles=legend_elements)
    # ax.margins(x=0, y=-.2)
    ax.set(xticks=[], yticks=[])
    

@st.cache
def load_model(path="data/topic-reduction.csv"):
    return pd.read_csv(root.parent / path)


st.title("Hierarchical topic reduction [top2vec]")
st.write("""
We used Top2Vec algorithm to automatically detect topics 
present in our ProQuest Deterrence dataset containing 26,525 non-empty documents. 

The model initially found 229 topics which is too much to interpret. Because of it, 
we decided to reduce the number of topics using top2vec's hierarchical topic reduction method.

To decide on the optimal number of topics, we visualize embeddings 
so that we could see clusters of documents""")

st.sidebar.markdown("## Controls")
st.sidebar.markdown("You can **change** the values to change the *chart*.")
num_topic = st.sidebar.slider("Number of topics", min_value=2, max_value=20, step=1)
method = st.sidebar.selectbox("Select plotting method", ("Matplotlib", "Datashader"))

df = load_model()
viz = matplotlib_points if method == "Matplotlib" else datashader_points
plot = viz(df, num_topic)
st.pyplot(plot)