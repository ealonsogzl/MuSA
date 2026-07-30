#---modules---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


#---colormaps---
#snow colormap -> credits: Devon Dunmire (devon.dunmire@buffalo.edu)
cmap1 = plt.cm.Blues
cmap2 = plt.cm.plasma
cmap3 = plt.cm.plasma
colors1 = cmap1(np.linspace(0.01, 1,250))
colors2 = cmap2(np.linspace(0.2, 1, 400))
colors3 = cmap3(np.linspace(0.8, 1, 100))
colors = np.vstack((colors1, colors2))
colors_pd = pd.DataFrame(colors)
colors_pd = colors_pd.rolling(window = 250, min_periods = 1).mean()
x = colors_pd.values
colors = np.vstack((colors_pd.values, colors3))
colors_pd = pd.DataFrame(colors)
colors_pd = colors_pd.rolling(window = 50, min_periods = 1).mean()
colors_snow=colors_pd.values
cmap_snow = mcolors.LinearSegmentedColormap.from_list('cmap_snow', colors_snow)

#---functions---
def mean_absolute_error(y_true, y_pred):
    '''function to calculate the mean absolute error between two arrays'''
    mask=~np.isnan(y_true) & ~np.isnan(y_pred)
    return np.mean(np.abs(y_true[mask] - y_pred[mask]))

def root_mean_squared_error(y_true, y_pred):
    '''function to calculate the root mean squared error between two arrays'''
    mask=~np.isnan(y_true) & ~np.isnan(y_pred)
    return np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2))

def Make2DHistogram(
        dfPreds:pd.DataFrame, 
        target:str="sdMeas", 
        preds:str="sdPred",
        includeZeros:bool=False):
    # remove rows with NaN values in the target or predictions
    dfPreds = dfPreds.dropna(subset=[target, preds])

    #make 2D histogram
    fig, ax=plt.subplots(figsize=(5,5))
    if not includeZeros:
        dfPreds = dfPreds[(dfPreds[target] != 0)]
    h = ax.hist2d(dfPreds[target], dfPreds[preds], 
                bins=np.linspace(0,6, 100), 
                cmin=0.001, 
                norm=mcolors.LogNorm(vmin=1, vmax=1000),
                cmap=cmap_snow)
    
    cax=fig.add_axes([0.57, 0.25, 0.3, 0.03])
    cbar=plt.colorbar(h[3], ax=ax, cax=cax, orientation="horizontal", extend="max")
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.set_title("Count [-]", fontsize=16)
    ax.plot([0,6], [0,6],lw=1.5,ls="--",c="darkred")
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='x', labelsize=18)

    corr=round(np.corrcoef(dfPreds[preds], 
                        dfPreds[target])[0, 1],
                2)
    rmse=round(root_mean_squared_error(dfPreds[preds],
                                        dfPreds[target]),
                2)
    mae=round(mean_absolute_error(dfPreds[target],
                                    dfPreds[preds]),
                2)
    bias=round(np.mean(dfPreds[target] - dfPreds[preds]),4)
    # Add performance metrics as text
    metrics = [f"R: {corr:.2f}", 
                f"RMSE: {rmse:.2f} m", 
                f"MAE: {mae:.2f} m",
                f"bias: {bias:.2f} m"]
                #f"R²: {r2:.2f}"]
    for j, metric in enumerate(metrics):
        t = ax.text(0.5, 5.7 - j * 0.5, metric, fontsize=18)
        t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='white'))
    #remove right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("Measured SD (m)", fontsize=18)
    ax.set_ylabel("Predicted SD (m)", fontsize=18)