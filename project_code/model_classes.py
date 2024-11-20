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
import project_code.utils as ut
from data_processing_classes import DataExploration
# pip install --upgrade nbformat
# pip install -U kaleido

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

class AprioriModel(DataExploration):
    @staticmethod
    def filter_nbre_transactions(df : pd.DataFrame, threshold_transactions : pd.DataFrame) -> pd.DataFrame:
        """
        Apply a threshold on the minimum number of transactions over the past 3 months for every SKU.  
        """
        df = df[df["SKU_TOTAL_TRANSACTION_1"]>=threshold_transactions]
        df = df[df["SKU_TOTAL_TRANSACTION_2"]>=threshold_transactions]
        return df
    
    def create_all_possible_pairs_sku(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a datasets with all the possible pairs of SKUs.
        SKU 1 in clomun 1, SKU 2 in column 2.
        Remove pairs with the same SKUs.
        """
        ## Create all possible pairs
        pairs = list(product(df["SKU"].sort_values(), repeat=2))

        # # Create a new DataFrame with the pairs
        pairs_df = pd.DataFrame(pairs, columns=["SKU_1", "SKU_2"])
        pairs_df["TRANSACTION_COUNT"] = 0
        
        ## Remove pairs with the same SKUs
        pairs_df = pairs_df[pairs_df["SKU_1"] != pairs_df["SKU_2"]]
        return pairs_df
    
    def create_tuple_skus_sold_together(self, df : pd.DataFrame) -> pd.DataFrame:
        """
        Find all the pairs of SKUs sold in the same basket.
        """
        ## Aggregate all the SKUs sold in the same basket into a list
        skus_per_transaction = df.groupby(["TRANSACTION_ID"]).agg({
            "SKU" : lambda x : set(list(x))
        }).reset_index()

        ## Sort the list of SKUs by SKU
        skus_per_transaction["SKU"] = skus_per_transaction.SKU.map(lambda x : sorted(x))

        ## Create all possible pairs of SKUs for every basket
        skus_per_transaction["PAIR_SKUS"] = skus_per_transaction["SKU"].map(lambda x : list(product(x, repeat=2)))
        return skus_per_transaction
    
    def count_pair_purchase(self, df_pairs_sku : pd.DataFrame, df_transactions_skus : pd.DataFrame) -> pd.DataFrame:
        """
        Count the number of times a pair of SKUs was sold together.
        Input:
            - Dataframe containing all the possible pairs of SKUs.
            - Dataframe containing the SKUs sold together for every transactions.     
        Output:
            - Dataframe containing all the possible pairs of SKUs.  
        """
        for _, row in df_transactions_skus.iterrows():
            list_pairs = row["PAIR_SKUS"]
            for _pair in list_pairs:
                sku_id1 = _pair[0]
                sku_id2 = _pair[1]
                # if sku_id1 < sku_id2:
                df_pairs_sku.loc[(df_pairs_sku["SKU_1"]==sku_id1) & (df_pairs_sku["SKU_2"]==sku_id2), "TRANSACTION_COUNT"] += 1

        return df_pairs_sku
    
    def add_skus_data(self, df_pairs : pd.DataFrame, df_skus : pd.DataFrame) -> pd.DataFrame:
        """
        Add statistics about both SKUs contained in every pair of a same basket.
        """
        ## Rename columns of the dataset containing unique SKUs, their cluster and statsitics
        skus_data1 = df_skus.copy()
        skus_data1.columns = [str(x)+"_1" for x in skus_data1.columns]

        ## Add statistics for SKU 1 using the suffixe 'SKU_1' 
        df_pairs = df_pairs.merge(skus_data1, on = "SKU_1", how = "left")

        ## Same for SKU 2
        skus_data2 = df_skus.copy()
        skus_data2.columns = [str(x)+"_2" for x in skus_data2.columns]
        df_pairs = df_pairs.merge(skus_data2, on = "SKU_2", how = "left")

        return df_pairs
    
    def flag_same_cluster(self, df : pd.DataFrame) -> pd.DataFrame:
        """
        Flag whether the SKUs contained in a pair have the same cluster.
        """
        df["SAME_CLUSTER"] = df.apply(lambda x : 1 if x.CLUSTER_NAME_1 == x.CLUSTER_NAME_2 else 0, axis = 1)
        return df
    
    def generate_features(self, df : pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature engineering and return for every pairs:
        - Confidence metric
        - Lift metric
        - Conviction  metric
        """
        df["CONFIDENCE"] = df["TRANSACTION_COUNT"]/df["SKU_TOTAL_TRANSACTION_1"]
        df["LIFT"] = df["CONFIDENCE"]/df["SUPPORT_CLUSTER_2"]
        df["CONVICTION"] = (1-df["SUPPORT_CLUSTER_2"])/(1-df["CONFIDENCE"])
        return df
    

    def correlation_skus_detection(self, df : pd.DataFrame) -> pd.DataFrame:
        """
        Apply filters to flag items whose sales are positively correlated (i.e.: measure of halo effect between the two).
        """
        df_filtered = AprioriModel.filter_nbre_transactions(df, ut.CORRELATION_THRESHOLDS["threshold_number_transactions"])
        ## Apply filtering using lift metric
        df_filtered = df_filtered[(df_filtered["LIFT"]>=ut.CORRELATION_THRESHOLDS["lift_threshold"])]

        ## Apply filtering using confidence metric
        df_filtered = df_filtered[(df_filtered["CONFIDENCE"]>=ut.CORRELATION_THRESHOLDS["confidence_threshold"])]

        ## Apply filtering using support metric for the pair of items
        df_filtered = df_filtered[(df_filtered["TRANSACTION_COUNT"]>=ut.CORRELATION_THRESHOLDS["support_pair_threshold"])]

        ## Apply filtering using conviction metric
        df_filtered = df_filtered[(df_filtered["CONVICTION"]>=ut.CORRELATION_THRESHOLDS["conviction_threshold"])]

        ## Only keep pairs from the same cluster
        df_filtered = df_filtered[(df_filtered["SAME_CLUSTER"]==1)]
        return df_filtered
    
    def find_top_3_upsells(self, df : pd.DataFrame) -> pd.DataFrame:
        """
        For every item find the top 3 most correlated other items.
        """
        return df.sort_values(by=["SKU_1", "LIFT"], ascending=[True, False]).groupby("SKU_1").head(3).reset_index(drop=True) 
    

class NeighborsGraph(AprioriModel):
    def find_nearest_neighbors_node(self, G) -> dict:
        """
        Find the nearest neighbors of all the nodes in the graph and the weight of the edge between the node and its neighbors
        with a degree of 1. 
        """
        ## Create a dictionnary with all the nodes of the graph as keys
        node_neighbor_weights = {_node: {}  for _node in G.nodes()}

        ## For every node of the graph:
        for _node in G.nodes():
            ## Find the nearest neighbors
            node_neighbors = G.neighbors(_node)    
            for _neighbor in node_neighbors:
                ## Return the value of the weight for the edge between the node and its neighbor
                edge_neighbor = G.get_edge_data(_node, _neighbor)
                
                ## Add the neighbor ID and the edge value to the dictionnary
                node_neighbor_weights[int(_node)][int(_neighbor)] = edge_neighbor["weight"]
        return node_neighbor_weights

    def order_neighbors_per_weight(self, node_neighbors_dict : dict) -> dict:
            """
            For every node, it orders the list of its nearest neighbors given the weight of the edge between them. 
            """
            for _node, _neighbors in node_neighbors_dict.items():
                node_neighbors_dict[_node] = dict(sorted(_neighbors.items(), key=lambda item: item[1], reverse = True))
            return node_neighbors_dict

    def find_n_plus_1_neighbors_node(self, G, node_neighbors_dict : dict) -> dict:
        """
        For every node, it goes through the list of its nearest neighbors ordered by importance (i.e.: the weight of the edge).
        It then finds their nearest neighbors (i.e.: they are the neighbors of the original node with a degree of 2).
        It orders the list of neighbors given their weight.
        It returns the original list of nodes with their neighbors with a degree of 2 ordered by their weight.
        """
        ## Create a dictionnary with all the nodes of the graph as keys
        node_neighbor_weights_n_1 = {_node: []  for _node in G.nodes()}

        ## Iter through all the nearest neighbors of every nodes
        for _node_0, _neighbors_1 in node_neighbors_dict.items():
            ## Create a dictionnary to add all the nodes with a of degree 2
            node_neighbor_weights_n_2 = {_node: {}  for _node in _neighbors_1.keys()}
            for _neighbor_1 in _neighbors_1.keys():
                ## Find the neighbors (degree 2) of the neighbor (degree 1)
                node_neighbors_2 = G.neighbors(_neighbor_1)    
                for _neighbor_2 in node_neighbors_2:
                    ## Save the neighbor with a degree of 2 and the weight of the edge shared with neighbor with a degree of 1
                    edge_neighbor = G.get_edge_data(_neighbor_1, _neighbor_2)
                    node_neighbor_weights_n_2[int(_neighbor_1)][int(_neighbor_2)] = edge_neighbor["weight"]

                node_neighbor_weights_n_2 = self.order_neighbors_per_weight(node_neighbor_weights_n_2)

                node_neighbor_weights_n_1[_node_0] += list(node_neighbor_weights_n_2[_neighbor_1].keys())
        return node_neighbor_weights_n_1

    def create_graph(self, df : pd.DataFrame):
        """
        Create a graph to represent the intereactions between all the SKUs.
        - Node: a unique SKU
        - Edge: the conviction score between Node 1 and Node 2 
        """
        G = nx.Graph()
        for _, row in df.iterrows():
            G.add_node(int(row["SKU_1"]), label=row["ITEM_DESCRIPTION_1"])
            G.add_node(int(row["SKU_2"]), label=row["ITEM_DESCRIPTION_2"])
            G.add_edge(int(row["SKU_1"]), int(row["SKU_2"]), weight=row["CONVICTION"])
        return G
    
    def remove_node_neighbors(self, node_neighbors_dict : dict) -> dict:
        """
        Remove the node ID of its neighbors with a degree of 2 list.
        """
        for _node, _neighbors in node_neighbors_dict.items():
            node_neighbors_dict[_node] = [x for x in _neighbors if x!=_node]
        return node_neighbors_dict
    
    def all_possible_neighbors(self, df_skus) -> pd.DataFrame:
        """
        Create a sparse dataset for every possible pair of SKUs.
        NEIGHBOR_RANKING is a score given to SKU 2 - neighbor of SKU 1 with a degree of 2.
        NEIGHBOR_RANKING = 1, 2 or 3, with 1 beign the best. 
        """
        df = self.create_all_possible_pairs_sku(df_skus)
        df["NEIGHBOR_RANKING"] = 0
        return df
    
    def remove_duplicated_neighbors(self, node_neighbors_dict : dict) -> dict:
        """
        Input:
            - node_neighbor_n_plus_one: Key: SKU, value: list of every neighbors with a degree of 2.
        Remove duplicated neighbors for every SKU.
        """
        for _node, _neighbors in node_neighbors_dict.items():
            node_neighbors_dict[_node] = list(dict.fromkeys(_neighbors))
        return node_neighbors_dict
    
    def find_best_neighbors_n_2(self, node_neighbors_dict : dict, neighbors_degree_2 : pd.DataFrame) -> pd.DataFrame:
        """
        Input:
            - node_neighbor_n_plus_one: Key: SKU, value: list of every neighbors with a degree of 2.
        For every SKU, return the 3 best neighbors with a degree of 2, ranked from 1 to 3.
        """
        for _node, _neighbors in node_neighbors_dict.items():
            count_iter = 1
            for node_neighbor in _neighbors:
                    if count_iter < 4:
                        neighbors_degree_2.loc[(neighbors_degree_2["SKU_1"]==_node) & (neighbors_degree_2["SKU_2"]==node_neighbor), "TRANSACTION_COUNT"] = 1
                        neighbors_degree_2.loc[(neighbors_degree_2["SKU_1"]==_node) & (neighbors_degree_2["SKU_2"]==node_neighbor), "NEIGHBOR_RANKING"] = count_iter
                        count_iter +=1
        return neighbors_degree_2[neighbors_degree_2["TRANSACTION_COUNT"]>0]
    
    def add_features_to_neighbors_n_2(self, df : pd.DataFrame, df_skus) -> pd.DataFrame:
        """
        Calculate features (lift, confidence etc.) for the new neighbors with a degree of 2.
        Flag the pairs with from the same cluster.
        """
        df = self.add_skus_data(df, df_skus) 
        df = self.generate_features(df)
        df = self.flag_same_cluster(df)
        return df
    
    
    def add_neighbors_n_2_to_n_1(self, df_pairs : pd.DataFrame, df_neighbors_n_2 : pd.DataFrame) -> pd.DataFrame:
            """
            Concatenate neighbors with a degree of 1 and a degree of 3.
            For every SKU, returns the closest:
                - Keep the N neighbors at a degree of 1 which made the cut
                - If N<3, keep the neighbors with a degree of 2 so that the maximum number of up-sell items per unique SKU is 3
            """
            df_pairs["NEIGHBOR_RANKING"] = 0
            pairs_df_top_neighbors = pd.concat([df_pairs, df_neighbors_n_2])
            return pairs_df_top_neighbors.sort_values(by=["SKU_1", "NEIGHBOR_RANKING"], ascending=[True, True]).groupby("SKU_1").head(3).reset_index(drop=True)
