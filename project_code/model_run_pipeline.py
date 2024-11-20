import pandas as pd
import numpy as np
import os
import pyvis
import networkx as nx
from matplotlib import pyplot as plt
from itertools import product
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import utils as ut
from data_processing_classes import DataExploration
from model_classes import AprioriModel, NeighborsGraph
# pip install --upgrade nbformat
# pip install -U kaleido

build_model = NeighborsGraph()

def run_model_pipeline():
    skus_data, transactions_data = build_model.run_data_preprocessing_piepeline()

    ## Generate revenue and transactions statistics per cluster
    cluster_purchases = build_model.total_transactions_per_cluster(transactions_data)

    ## Merge SKUs and Clusters datasets
    skus_data = build_model.merge_sku_cluster_statistics(skus_data, cluster_purchases)

    ## Calculate the Support metric of every item givven the total number of transactions per cluster
    skus_data = build_model.calculate_support_sku(skus_data)

    ##Create all possible pairs of SKUs
    pairs_df = build_model.create_all_possible_pairs_sku(skus_data)

    ## Find all the pairs of SKUs sold in the same basket
    skus_per_transaction = build_model.create_tuple_skus_sold_together(transactions_data)

    ## Count the number of times a pair of SKUs was sold together
    pairs_df = build_model.count_pair_purchase(pairs_df, skus_per_transaction)

    ## Merge SKUs and pairs of SKUs datasets
    pairs_df = build_model.add_skus_data(pairs_df, skus_data)

    ## Flag whether a pair of SKUs is from the same cluster
    pairs_df = build_model.flag_same_cluster(pairs_df)

    ## Generate metrics (Lift, Confidence etc.)
    pairs_df = build_model.generate_features(pairs_df)

    ## Apply thresholds to the metrics to flag correlated items
    pairs_df_filtred = build_model.correlation_skus_detection(pairs_df)

    ## Return 3 items to up-sell per unique SKU
    pairs_df_top3 = build_model.find_top_3_upsells(pairs_df_filtred)   

    return pairs_df_top3, skus_data, transactions_data

def generate_upsell_reco(df : pd.DataFrame) -> pd.DataFrame:
    """
    Format the dataframe to generate a file containing SKUs with the other items to up-sell. 
    """
    ## Order the items so that 'rec 1' is better than 'rec 2', better than 'rec 3'
    pairs_df_top_neighbors_sorted = df.sort_values(by=["SKU_1", "NEIGHBOR_RANKING", "CONVICTION"], ascending=[True, True, False])

    ## Group by SKU 1 and collect sorted SKU 2 as a list
    pairs_df_top_neighbors_sorted = pairs_df_top_neighbors_sorted.groupby("SKU_1")["SKU_2"].apply(list).reset_index()

    ## Create a column for every up-sell item
    df_reco = pairs_df_top_neighbors_sorted.join(
        pd.DataFrame(pairs_df_top_neighbors_sorted["SKU_2"].tolist(), index=pairs_df_top_neighbors_sorted.index, columns=["rec 1", "rec 2", "rec 3"])
    )

    ## Drop and rename columns before saving file
    df_expanded = df_reco.drop(columns=["SKU_2"])
    # df_expanded = df_expanded.fillna(0)
    # df_expanded = df_expanded.astype(int)

    df_expanded = df_expanded.rename(columns = {"SKU_1" : "sku"})

    ## Save file
    df_expanded.to_csv(f"{ut.CURRENT_DIRECTORY}/results/upsell_results.csv", index=False)
    return df_expanded

if __name__ == "__main__":
    ## Import SKUs and Transactions datasets
    pairs_df_top3, skus_data, transactions_data = run_model_pipeline()

    ## Create graph using pairs of SKUs and metrics
    G = build_model.create_graph(pairs_df_top3)

    ## Find the nearest neighbor (with a degree of 1) for every SKU
    node_neighbor_weights = build_model.find_nearest_neighbors_node(G)

    ## Order every nearest neighbour given a metric
    node_neighbor_weights_order = build_model.order_neighbors_per_weight(node_neighbor_weights)

    ## Find the neighbors with a degree of 2 for every SKU
    node_neighbor_n_plus_one = build_model.find_n_plus_1_neighbors_node(G, node_neighbor_weights_order)

    ## Remove the node from the lost of its neighbors
    node_neighbor_n_plus_one = build_model.remove_node_neighbors(node_neighbor_n_plus_one)

    ## Remove duplicated in the list of neighbors
    node_neighbor_n_plus_one = build_model.remove_duplicated_neighbors(node_neighbor_n_plus_one)

    ## Create a sparse dataset with all the SKU pairs possible
    neighbors_degree_2 = build_model.all_possible_neighbors(skus_data)

    ## Find the best 3 neighbors with a degree of 2
    neighbors_degree_2 = build_model.find_best_neighbors_n_2(node_neighbor_n_plus_one, neighbors_degree_2)

    ## Calculate the metrics between a node and its neighbors with a degree of 2
    neighbors_degree_2 = build_model.add_features_to_neighbors_n_2(neighbors_degree_2, skus_data)

    ## Add neighbors with a degree of 2 to the neighbors with a degree of 1 (existing top 3 up-sell items for every SKU)
    ## Fill the top 3 with neighbors with a degree of 2 if top 3 incomplete
    pairs_df_top_neighbors = build_model.add_neighbors_n_2_to_n_1(pairs_df_top3, neighbors_degree_2)

    df_expanded = generate_upsell_reco(pairs_df_top_neighbors)
    print(df_expanded)