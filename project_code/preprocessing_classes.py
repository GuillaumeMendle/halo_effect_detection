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
# pip install --upgrade nbformat
# pip install -U kaleido

class DataPreprocessing:
    def import_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return the SKUs and Transactions datasets.  
        """
        skus_data = pd.read_csv(f"{ut.CURRENT_DIRECTORY}/data/skus.csv", delimiter = ",")
        transactions_data = pd.read_csv(f"{ut.CURRENT_DIRECTORY}/data/transactions.csv", delimiter = ",")
        return skus_data, transactions_data

    def merge_skus_transactions(self, df_skus : pd.DataFrame, df_transactions : pd.DataFrame) -> pd.DataFrame:
        """
        Merge SKUs and Transactions datasets.  
        """
        return df_transactions.merge(df_skus, on = "SKU", how = "left")
    
    def add_transaction_details_skus(self, df_skus : pd.DataFrame, df_baskets : pd.DataFrame) -> pd.DataFrame:
        """
        Add the Total Revenue and Total Number of transactions for every item.
        """
        return df_skus.merge(df_baskets, on = "SKU", how = "left")
    
    def calculate_items_bought(self, df : pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the number of times an item appears in a basket and sum the total revenue generated. 
        """
        return df.groupby(["SKU"]).agg(
            SKU_TOTAL_TRANSACTION = ("TRANSACTION_ID", "count"),
            SKU_TOTAL_REVENUE = ("REVENUE", "sum")
    ).reset_index()

    def remove_anomalies(self, df : pd.DataFrame) -> pd.DataFrame:
        """
        Remove negative Quantity and Revenue for data consistency.
        """
        df = df[df["QTY"]>0]
        df = df[df["REVENUE"]> 0]
        return df
    
    def create_cluster_id(self, row : pd.Series) -> str:
        """ 
        Create a cluster name for every item given Department, Category, Subcategory1, etc.
        """
        department_name = row["DEPARTMENT"]
        category_name = row["CATEGORY"]
        subcategory_1_name = row["SUBCATEGORY1"]

        cluster_name = department_name
        if department_name == "Weighed":
            cluster_name = category_name
        if department_name == "Services":
            cluster_name = subcategory_1_name
        if cluster_name == "Bird" or cluster_name == "Wildbird" or cluster_name == "Dom.Bird":
            cluster_name = "Bird"
        return cluster_name

    def run_data_preprocessing_piepeline(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Import the datasets and run the whole preprocessing pipeline. 
        """
        skus_data, transactions_data = self.import_data()
        skus_data["CLUSTER_NAME"] = ""
        skus_data["CLUSTER_NAME"] = skus_data.apply(lambda x : self.create_cluster_id(x), axis = 1)
        transactions_data = self.merge_skus_transactions(skus_data, transactions_data)
        transactions_data = self.remove_anomalies(transactions_data)
        df_item_purchases = self.calculate_items_bought(transactions_data)
        skus_data = self.add_transaction_details_skus(skus_data, df_item_purchases)
        return skus_data, transactions_data
    

class DataExploration(DataPreprocessing):
    def find_missing_SKU_transactions(self, df_skus : pd.DataFrame, df_transactions : pd.DataFrame) -> pd.DataFrame:
        """
        Return missing SKUs in the transactions dataset. 
        """
        list_skus = {x for x in df_skus["SKU"].unique() if x not in df_transactions["SKU"].unique()}
        return df_skus[df_skus["SKU"].isin(list_skus)]
    
    def count_items_sold_transaction(self, df : pd.DataFrame) -> pd.DataFrame:
        """
        Count the number of items sold per basket for each transaction. 
        """
        return df.groupby(["TRANSACTION_ID"]).agg({
            "SKU" : lambda x : x.nunique()
        }).reset_index().value_counts("SKU")

    def plot_distribution_items_transaction(self, df : pd.DataFrame) -> None:
        """
        Save the distribution of numbe of items in a basket. 
        """
        plt.figure(figsize = (8,6))
        plt.bar(df.index, df.values,color='skyblue', edgecolor='black', alpha=0.8)
        plt.title("Distribution of the number of items in a basket", fontsize=12, fontweight='bold')
        plt.xlabel("Number of items per basket", fontsize=12)
        plt.ylabel("Number of basket with X items", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.savefig(f"{ut.CURRENT_DIRECTORY}/data/nbre_items_per_transaction.png")
        plt.close()

    def plot_hierarchical_categories(self, df : pd.DataFrame) -> None:
        """
        Save plot of the hierarchy within the SKUs data. Departments -> Categories -> etc.
        """
        hierarchical_categories = df[["DEPARTMENT", "CATEGORY"]].drop_duplicates()
        fig = px.treemap(hierarchical_categories, path=["DEPARTMENT", "CATEGORY"])
        fig.write_image(f"{ut.CURRENT_DIRECTORY}/data/department_category_tree.png")


    def share_revenue_items_sold(self, df : pd.DataFrame, threshold_transactions : int) -> float:
        """
        Return percentage of revenue one keeps after dropping low-frequency items. 
        """
        return df[df["SKU_TOTAL_TRANSACTION"] >= threshold_transactions]["SKU_TOTAL_REVENUE"].sum()/df["SKU_TOTAL_REVENUE"].sum()
    