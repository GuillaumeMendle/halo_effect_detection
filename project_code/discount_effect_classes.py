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

class DiscountEffect:
    def discount_flag_and_calculation(self, df : pd.DataFrame) -> pd.DataFrame:
        """
        Flag if a discount is applied on the item purchased for a transaction.
        Calculate the percentage of discount applied. 
        """
        ## Flag if a discount is applied to the sale
        df["DISCOUNT_APPLIED"] = df["DISCOUNT"].map(lambda x : 0 if x==0 else 1)

        ## Calculate the percentage of the discount applied
        df["DISCOUNT_VALUE"] = 1-df["SALE_PRICE_EX_VAT"]/(df["SALE_PRICE_EX_VAT"] + df["DISCOUNT"])
        return df
    
    def flag_discounted_SKU(self, df : pd.DataFrame) -> pd.DataFrame:
        """
        Return SKUs which got sold discounted and not discounted over the past 3 months.
        """
        discount_df = df.groupby(["SKU"]).agg(
                DISCOUNT_NO_DISCOUNT = ("DISCOUNT_APPLIED", lambda x : x.nunique())
        ).reset_index()

        return df.merge(discount_df, on = "SKU", how = "left")
    
    def filter_skus_per_discount_value(self, df : pd.DataFrame, threshold_discount_min : int) -> pd.DataFrame:
        """
        Keep transactions with a threshold_discount_min discount < threshold_discount_max and transactions with no discount.
        It will be used for further analysis to compare the effect of the discount percentage.
        """

        threshold_discount_max = threshold_discount_min+0.05
        ## Find SKUs that have been sold with at normal price and discounted price lower than the threshold
        sku_discount = df[df["DISCOUNT_VALUE"]>threshold_discount_min]["SKU"]

        ## Only keep the transaction for those SKUs
        df = df[(df["SKU"].isin(sku_discount))]

        ## Return only the transactions with a discount < threshold or no discount
        return df[(df["DISCOUNT_VALUE"].between(threshold_discount_min, threshold_discount_max)) | (df["DISCOUNT_APPLIED"]==0)]
    
    def measure_sales_volume_per_transaction(self, df : pd.DataFrame) -> pd.DataFrame:
        """
        1. Find the volume of quantity sold with and without discount.
        2. Calculate the ratio of quantity sold per transaction.
        3. Return the average of the calculated ratio for transaction with and without discount for comparison.
        """
        ## Find the volume of quantity sold with and without discount
        qty_df = df.groupby(["SKU", "DISCOUNT_APPLIED"]).agg(
                QTY_SOLD = ("QTY", "sum"),
                NBRE_TRANSACTIONS = ("QTY", "count")
        )

        ## Calculate the ratio of quantity sold per transaction
        qty_df["RATIO_QTY_SOLD"] = qty_df["QTY_SOLD"]/qty_df["NBRE_TRANSACTIONS"]
        qty_df = qty_df.reset_index()

        ## Return the average of the calculated ratio for transaction with and without discount
        return qty_df[qty_df["DISCOUNT_APPLIED"]==0]["RATIO_QTY_SOLD"].mean(), qty_df[qty_df["DISCOUNT_APPLIED"]==1]["RATIO_QTY_SOLD"].mean()
    
    def measure_impact_discount(self, df : pd.DataFrame, threshold_discount_max : float) -> pd.DataFrame:
        """
        Return the average ratio quantity/transactions for items sold without discount
        and items sold after applying a promotion lower than a threshold. 
        """
        all_threshold_discount = []
        all_ratio_no_discount = []
        all_ratio_discount = []
        
        ## Measure impact of discount for a range of thresholds
        for threshold_discount in np.arange(0, threshold_discount_max, 0.05):
            df1 = df.copy()
            df1 = df1[df1["DISCOUNT_NO_DISCOUNT"]==2]
            transactions_data_discount_filtered = self.filter_skus_per_discount_value(df1, threshold_discount)
            ratio_without_discount, ratio_with_discount = self.measure_sales_volume_per_transaction(transactions_data_discount_filtered)

            all_threshold_discount.append(threshold_discount)
            all_ratio_no_discount.append(ratio_without_discount)
            all_ratio_discount.append(ratio_with_discount)
        return pd.DataFrame({"DISCOUNT_VALUE" : all_threshold_discount, "SALES_RATIO_NO_DISCOUNT" : all_ratio_no_discount, "SALES_RATIO_WITH_DISCOUNT" : all_ratio_discount})

    def plot_impact_discount(self, df : pd.DataFrame) -> None:
        """
        Generate a graph comparing the quantity sold per transaction with and without discount applied on the same items.
        """
        plt.figure(figsize = (8,6))
        plt.plot(df["DISCOUNT_VALUE"], df["SALES_RATIO_NO_DISCOUNT"], label = "QUANTITY SOLD PER TRANSACTION - NO DISCOUNT")
        plt.plot(df["DISCOUNT_VALUE"], df["SALES_RATIO_WITH_DISCOUNT"], label = "QUANTITY SOLD PER TRANSACTION - WITH DISCOUNT")
        plt.xticks(df["DISCOUNT_VALUE"])
        plt.yticks()
        plt.legend()
        plt.xlabel("Percentage of discount applied")
        plt.ylabel("Average quantity sold per transaction")
        plt.title("Average quantity sold per transaction - with/without discount")
        plt.grid()
        plt.savefig(f"{ut.CURRENT_DIRECTORY}/plots_graph/impact_discount_on_quantity.png")
        plt.close()