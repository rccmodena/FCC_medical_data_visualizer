#!/usr/bin/env python3
"""
Author: Rudi César Comiotto Modena
Email: rudi.modena@gmail.com
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
filename = "medical_examination.csv"
df = pd.read_csv(filename)

# Add 'overweight' column
df['overweight'] = df.apply(lambda row : 1 if (row['weight'] / (row['height'] / 100) ** 2) > 25  else 0, axis=1)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df['cholesterol'] = df['cholesterol'].apply(lambda var : 1 if var > 1 else 0)
df['gluc'] = df['gluc'].apply(lambda var : 1 if var > 1 else 0)

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df, id_vars=['id', "cardio"], value_vars=sorted(['cholesterol', 'gluc', 'smoke', 'alco', 'active','overweight']))

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    #df_cat = None

    # Draw the catplot with 'sns.catplot()'
    g = sns.catplot(x="variable", col="cardio", hue="value", data=df_cat, kind="count")
    g.set(ylabel='total', xlabel='variable')
    fig = g.fig

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    sns.reset_orig()
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df[(df['ap_lo'] <= df['ap_hi'])
    & (df['height'] >= df['height'].quantile(0.025))
    & (df['height'] <= df['height'].quantile(0.975))
    & (df['weight'] >= df['weight'].quantile(0.025))
    & (df['weight'] <= df['weight'].quantile(0.975))]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11, 9))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, mask=mask, center=0, vmin=-0.2, vmax=0.5, xticklabels=True, yticklabels=True, square=True, cmap="icefire", annot=True, fmt=".1f", linewidths=2, cbar_kws={'shrink': 0.6}, annot_kws={'size': 10}, ax=ax)

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    sns.reset_orig()
    return fig
