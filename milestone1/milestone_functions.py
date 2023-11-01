import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns


def ldl(value):
    if '<' in value:
        # Extract the numerical value without '<' and convert it to float
        result = float(re.sub('<(\d+)', r'\1', value))
        # Calculate 'ldl' based on the extracted value
        ldl_var = result / 2
        return ldl_var
    else:
        # If the value does not contain '<', return the original value
        return float(value)
    
def udl(value):
    if '>' in value:
        # Extract the numerical value without '<' and convert it to float
        result = float(re.sub('>(\d+)', r'\1', value))
        # Calculate 'ldl' based on the extracted value
        udl_var = result * 1.5
        return udl_var
    else:
        # If the value does not contain '<', return the original value
        return float(value)
    

    # function for the correlation matrix

def plot_correlation_heatmap(df, name = 'df'):

    plt.figure(figsize=(15,12))

    corr_mat_train = df.corr()

    # Create a mask to hide the upper triangle
    mask = np.triu(corr_mat_train)

    # Heatmap plotting
    sns.heatmap(corr_mat_train, cmap='coolwarm',
                linewidths=.5, mask=mask,
                cbar_kws={"shrink": 0.7})

    plt.title(f"Correlation Matrix for {name} Data", fontsize=16)

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, ha='right', fontsize=10)

    plt.show()


    # function for the histogram plots

def plot_numeric_feature_histograms(df):


    # Calculate the number of subplots needed
    numeric_feature_names = df.columns
    num_features = len(numeric_feature_names)
    num_cols = 3
    num_rows = (num_features + num_cols - 1) // num_cols

    # Create subplots with specified rows and columns
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))

    # Loop through numeric features and plot histograms
    for i, feature_name in enumerate(numeric_feature_names):
        row = i // num_cols
        col = i % num_cols
        sns.histplot(df[feature_name], kde=True, ax=axs[row, col], color='orange', bins=30)
        axs[row, col].set_title(f'Histogram for {feature_name}')
        axs[row, col].set_xlabel(feature_name)
        axs[row, col].set_ylabel('Frequency')

    # Hide empty subplots, if any
    for j in range(num_features, num_rows * num_cols):
        fig.delaxes(axs.flatten()[j])

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plots
    plt.show()

    # function for the scatter plots

def plot_scatterplots(df):
    X = df.iloc[:, :-1]
    y = df['Zn (ppm)']

    # Calculate the number of subplots needed
    num_rows = len(df.columns) // 3 + (len(df.columns) % 3 > 0)
    fig, axs = plt.subplots(num_rows, 3, figsize=(15, 3 * num_rows))

    # Flatten the axs array for easy indexing
    axs = axs.flatten()

    # Loop through feature names and plot scatterplots
    for i, feature_name in enumerate(X.columns):
        axs[i].scatter(X[feature_name], y, s=10)
        axs[i].set_title(f'Scatterplot for {feature_name} and Zn (ppm)')
        axs[i].set_xlabel(feature_name)
        axs[i].set_ylabel('Zn (ppm)')

    # Remove empty subplots, if any
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    # Adjust layout
    plt.tight_layout()

    # Show the plots
    plt.show()