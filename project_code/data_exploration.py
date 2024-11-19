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
from preprocessing_classes import DataExploration
# pip install --upgrade nbformat
# pip install -U kaleido

min_transactions = 6

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

