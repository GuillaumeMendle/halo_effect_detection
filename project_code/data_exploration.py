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
from discount_effect_classes import DiscountEffect

min_transactions = 4

if __name__ == "__main__":
    data_exploration = DataExploration()

    skus_data, transactions_data = data_exploration.run_data_preprocessing_piepeline()
    print("Average number of transactions per SKU: {}\nMedian: {}".format(round(skus_data['SKU_TOTAL_TRANSACTION'].mean(), 2), round(skus_data['SKU_TOTAL_TRANSACTION'].median(), 2)))
    
    data_exploration.plot_hierarchical_categories(skus_data)
    data_exploration.find_missing_SKU_transactions(skus_data, transactions_data)
    sku_transactions_data = data_exploration.count_items_sold_transaction(transactions_data)
    print(f"Total percentage of baskets containing at least 2 items: {round(1-sku_transactions_data.loc[1]/sku_transactions_data.sum(), 2)*100}%")
    data_exploration.plot_distribution_items_transaction(sku_transactions_data)
    remaining_revenue = data_exploration.share_revenue_items_sold(skus_data, min_transactions)
    print(f"Total percentage of revenue left after dropping items with less than {min_transactions} transactions: {round(remaining_revenue, 2)}%")

    ## Measure impact of promotion on quantity sold
    discount_effect = DiscountEffect()
    transactions_data_discount = transactions_data.copy()

    ## Flag when an item is discounted in for every transaction
    transactions_data_discount = discount_effect.discount_flag_and_calculation(transactions_data_discount)
    discount_or_no_df = transactions_data_discount['DISCOUNT_APPLIED'].value_counts()
    print(f"Distribution of discount/no discount in transactions: {round(discount_or_no_df[1]/discount_or_no_df.sum()*100,2)}%")

    ## Return SKUs which got sold discounted and not discounted over the past 3 months
    transactions_data_discount = discount_effect.flag_discounted_SKU(transactions_data_discount)

    ## Return the average ratio quantity/transactions for items sold without discount
    ## and items sold after applying a promotion lower than a threshold 
    df_impact_discount = discount_effect.measure_impact_discount(transactions_data_discount, 0.75)

    ## Generate a graph comparing the quantity sold per transaction with and without discount applied on the same items.
    discount_effect.plot_impact_discount(df_impact_discount)
