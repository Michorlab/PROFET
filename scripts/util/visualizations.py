## Static plot function for neural GPA

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import matplotlib.colors as mcolors
import os
from sklearn.cluster import KMeans
from sklearn import decomposition
import matplotlib.lines as mlines
import matplotlib.animation as animation
from IPython.display import Image
import pandas as pd
import torch # for W2 calculation purpose
from geomloss import SamplesLoss
from sklearn.metrics.pairwise import euclidean_distances

def W2(X, Y):
    X = torch.from_numpy(X).type(torch.float32)
    Y = torch.from_numpy(Y).type(torch.float32)
    return SamplesLoss(loss='sinkhorn', p=2)(X, Y).numpy()

 # compare W2 error with intermediate timepoints
def generate_W2distance_plot(full_matrix,time_label,projected_matrix, days, intermediate_days, X1_trpts, physical_dt, img_src, d_red, dimension_reduction, colors):
    # random sample color
    time_points = sorted(set(time_label))
    if colors == {}:
        np.random.seed(127)
        for day in time_points:
            colors[day] = np.random.random(3)

    all_ts = np.sort(days + intermediate_days)

    argmin_w2_s = []
    for day in all_ts:
        reference_mat = full_matrix[time_label==day,:] if dimension_reduction == False else projected_matrix[time_label==day,:d_red]
        w2_s = [W2(X1_trpt, reference_mat).item() for X1_trpt in X1_trpts]
        plt.plot(np.arange(0, len(X1_trpts)) * physical_dt, w2_s, color = colors[day], label = "Day" + str(day))
        argmin_w2_s.append(np.argmin(w2_s) * physical_dt)
        
    plt.legend()    
    print("minimum values are obtained at x=", argmin_w2_s)
    plt.title(r'$W_2(P_n, Q_{day})$')  
    plt.savefig(img_src)
    plt.show()
    

def generate_static_trajectory_plots_three_timepoints(pca,physical_dt, days, intermediate_days, X1_trpts, mats, d_red=26, output_file_with_snapshots=None, output_file_without_snapshots=None):
    """
    Generate two static trajectory plots:
    1. With snapshots from X1_trpts using a color gradient.
    2. Without snapshots, showing only main time points.
    """
    
    # Define color gradient for snapshots
    num_snapshots = len(X1_trpts)
    colormap = cm.viridis  # Can change to "plasma", "inferno", etc.
    snapshot_colors = [colormap(i / num_snapshots) for i in range(num_snapshots)]

    # Rescale time values for the color bar
    time_values = np.linspace(0, physical_dt * num_snapshots, num_snapshots)

    # Create a normalization object for the color mapping
    norm = mcolors.Normalize(vmin=time_values.min(), vmax=time_values.max())
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])  # Needed for color bar

    source_t, middle_t, target_t = days[0], days[1], days[-1]
    
    # Define colors for time points
    color_map = {
        source_t: '#1f77b4',  # Blue
        intermediate_days[0]: '#2ca02c',  # Green
        middle_t: '#ff7f0e',  # Orange
        intermediate_days[1]: '#8c564b',  # Brown
        target_t: '#d62728'  # Red
    }

    # **Plot 1: With Snapshots**
    fig1, ax1 = plt.subplots(figsize=(8, 6))

    # Plot source, intermediates, and target
    X1_vis = pca.transform(mats[source_t])
    Xm_vis = pca.transform(mats[middle_t])
    X2_vis = pca.transform(mats[target_t])
    ax1.scatter(X1_vis[:, 0], X1_vis[:, 1], color=color_map[source_t], alpha=1.0, s=12, zorder = 10, label=f'Time {source_t} (Training Data)')
    ax1.scatter(Xm_vis[:, 0], Xm_vis[:, 1], color=color_map[middle_t], alpha=1.0, s=12, zorder = 10, label=f'Time {middle_t} (Training Data)')
    ax1.scatter(X2_vis[:, 0], X2_vis[:, 1], color=color_map[target_t], alpha=1.0, s=12, zorder = 10, label=f'Time {target_t} (Training Data)')

    # Plot intermediate time points
    for t in intermediate_days:
        X_intermediate_vis = pca.transform(mats[t])
        ax1.scatter(X_intermediate_vis[:, 0], X_intermediate_vis[:, 1], color=color_map[t], facecolors='none', edgecolors=color_map[t], linewidths=1.2, alpha=1.0, s=15, zorder = 20,  label=f'Time {t} (Test Data)')

    # Plot snapshots from X1_trpts with a color gradient
    for i, X1_trpt in enumerate(X1_trpts):
        if np.isnan(X1_trpt).any():
            continue
        X1_hat_vis = X1_trpt
        ax1.scatter(X1_hat_vis[:, 0], X1_hat_vis[:, 1], color=snapshot_colors[i], alpha=0.75, s=5, zorder = 1)

    
    # Add a small color bar inside the plot
    cax = ax1.inset_axes([1.02, 0.2, 0.03, 0.6])  # [x, y, width, height] (relative position)
    
    # Create the colorbar with increased size
    cbar = plt.colorbar(sm, cax=cax)
    
    # Set manual tick positions
    cbar.set_ticks(np.linspace(0, 4, 5))  # Ensures ticks at 0, 1, 2, 3, 4
    
    # Optional: Explicitly set tick labels if needed
    cbar.set_ticklabels([0, 1, 2, 3, 4])  
    
    # Increase colorbar label font size
    cbar.set_label("Time", fontsize=20)  
    
    # Increase colorbar tick font size
    cbar.ax.tick_params(labelsize=20)

    # Adjust colorbar thickness
    #cbar.ax.set_aspect(20)  # Increase aspect ratio to make it thicker
   
    # Set labels and title
    ax1.set_xlabel("PC 1", fontsize = 20)
    ax1.set_ylabel("PC 2", fontsize = 20)
    ax1.tick_params(axis='both', which='major', labelsize=20)  # Increase tick sizes
    #ax1.legend(loc='upper right', fontsize= 24)
    ax1.set_title("")

    # Save or show the plot
    if output_file_with_snapshots:
        plt.savefig(output_file_with_snapshots, dpi=300, bbox_inches='tight')
        print(f"Static trajectory plot WITH snapshots saved to {output_file_with_snapshots}")
        plt.close(fig1)
    else:
        plt.show()

    # **Plot 2: Without Snapshots**
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    ax2.scatter(X1_vis[:, 0], X1_vis[:, 1], color=color_map[source_t], alpha=1.0, s=12, zorder = 10, label=f'Time {source_t} (Training Data)')
    ax2.scatter(Xm_vis[:, 0], Xm_vis[:, 1], color=color_map[middle_t], alpha=1.0, s=12, zorder = 10, label=f'Time {middle_t} (Training Data)')
    ax2.scatter(X2_vis[:, 0], X2_vis[:, 1], color=color_map[target_t], alpha=1.0, s=12, zorder = 10, label=f'Time {target_t} (Training Data)')

    # Plot intermediate time points
    for t in intermediate_days:
        X_intermediate_vis = pca.transform(mats[t])
        ax2.scatter(X_intermediate_vis[:, 0], X_intermediate_vis[:, 1], color=color_map[t], facecolors='none', edgecolors=color_map[t], linewidths=1.2, alpha=1.0, s=15, zorder = 20,  label=f'Time {t} (Test Data)')

    # Set labels and title
    ax2.set_xlabel("PC 1", fontsize = 20)
    ax2.set_ylabel("PC 2", fontsize = 20)
    ax2.tick_params(axis='both', which='major', labelsize=20)  # Increase tick sizes
    #ax2.legend(loc='upper right', fontsize='small')
    ax2.set_title("")

    # Save or show the plot
    if output_file_without_snapshots:
        plt.savefig(output_file_without_snapshots, dpi=300, bbox_inches='tight')
        print(f"Static trajectory plot WITHOUT snapshots saved to {output_file_without_snapshots}")
        plt.close(fig2)
    else:
        plt.show()


    
    # Extract legend elements
    handles, labels = ax1.get_legend_handles_labels()
    
    # Extract numeric values from "Time X (Input Data)" and "Time X (Test Data)"
    time_labels = []
    for label in labels:
        try:
            time_value = int(label.split(" ")[1])  # Extract the numerical value after "Time"
            time_labels.append((time_value, label))  # Store (time, label) pairs
        except ValueError:
            time_labels.append((float('inf'), label))  # Place non-time labels at the end
    
    # Sort legend by time values
    time_labels.sort(key=lambda x: x[0])  # Sort by the extracted numeric value
    sorted_labels = [item[1] for item in time_labels]
    sorted_handles = [handles[labels.index(label)] for label in sorted_labels]
    
    # **Increase marker size in legend**
    for handle in sorted_handles:
        if isinstance(handle, plt.Line2D):  # Ensure we're modifying scatter markers
            handle.set_markersize(30)  # Adjust marker size
    
    # Create a separate figure for the legend
    fig_legend, ax_legend = plt.subplots(figsize=(10, 2))  # Adjust size as needed
    ax_legend.axis("off")  # Remove axes
        
    # Create legend with larger markers for scatter plots
    legend = ax_legend.legend(
        sorted_handles, sorted_labels, fontsize=24, loc='center',
        ncol=len(sorted_labels), markerscale=4  # Increase scatter marker size
    )
    
    # Save the legend separately
    # legend_path = os.path.join(result_dir, "legend_only.png")
    # fig_legend.savefig(legend_path, bbox_inches="tight")
    plt.close(fig_legend)  # Close the legend figure
    
    #print(f"Legend saved separately at: {legend_path}")

## Static plot function for simple GPA (Sample 1 - EMT data)

                    
def generate_static_trajectory_plots_two_timepoints(pca,physical_dt,days, intermediate_days, X1_trpts, mats, d_red=26, output_file_with_snapshots=None, output_file_without_snapshots=None, output_file_snapshots_only=None):
    """
    Generate two static trajectory plots:
    1. With snapshots from X1_trpts using a color gradient.
    2. Without snapshots, showing only main time points.
    """

    
    # Define color gradient for snapshots
    num_snapshots = len(X1_trpts)
    colormap = cm.viridis  # Can change to "plasma", "inferno", etc.
    snapshot_colors = [colormap(i / num_snapshots) for i in range(num_snapshots)]

    # Rescale time values for the color bar
    time_values = np.linspace(0, physical_dt * num_snapshots, num_snapshots)

    # Create a normalization object for the color mapping
    norm = mcolors.Normalize(vmin=time_values.min(), vmax=time_values.max())
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])  # Needed for color bar

   
    source_t = days[0]
  
    target_t = days[1]
    # target_t = 4
    # Define colors for time points
    color_map = {
        source_t: '#1f77b4',  # Blue
        #intermediate_days[0]: '#ff7f0e',  # Orange
        target_t: '#d62728'  # Red
    }

    # **Plot 1: With Snapshots**
    fig1, ax1 = plt.subplots(figsize=(8, 6))

    # Plot source, intermediates, and target
    X1_vis = pca.transform(mats[source_t])
    #Xm_vis = pca.transform(mats[middle_t])
    X2_vis = pca.transform(mats[target_t])
    #ax1.scatter(X1_vis[:, 0], X1_vis[:, 1], facecolors='none', edgecolors=color_map[source_t], linewidths=0.5, alpha=1.0, s=20, zorder=10, label=f'Time {source_t}')
    #ax1.scatter(X2_vis[:, 0], X2_vis[:, 1], facecolors='none', edgecolors=color_map[target_t], linewidths=0.5, alpha=1.0, s=20, zorder=10, label=f'Time {target_t}')


    # Plot intermediate time points
    for t in intermediate_days:
        X_intermediate_vis = pca.transform(mats[t])
        ax1.scatter(X_intermediate_vis[:, 0], X_intermediate_vis[:, 1], color=color_map[t], facecolors='none', edgecolors=color_map[t], linewidths=1.0, alpha=0.75, s=10, zorder = 20,  label=f'Day {t} (Test Data)')

    # Plot snapshots from X1_trpts with a color gradient
    for i, X1_trpt in enumerate(X1_trpts):
        if np.isnan(X1_trpt).any():
            continue
        X1_hat_vis = X1_trpt
        ax1.scatter(X1_hat_vis[:, 0], X1_hat_vis[:, 1], color=snapshot_colors[i], alpha=0.75, s=2, zorder = 1)

    # Add a small color bar inside the plot
    cax = ax1.inset_axes([1.02, 0.2, 0.03, 0.6])  # [x, y, width, height] (relative position)
    
    # Create the colorbar with increased size
    cbar = plt.colorbar(sm, cax=cax)
    
    # Set manual tick positions
    cbar.set_ticks(np.linspace(0, 4, 5))  # Ensures ticks at 0, 1, 2, 3, 4
    
    # Optional: Explicitly set tick labels if needed
    cbar.set_ticklabels([0, 1, 2, 3, 4])  
    
    # Increase colorbar label font size
    cbar.set_label("Time", fontsize=20)  
    
    # Increase colorbar tick font size
    cbar.ax.tick_params(labelsize=20)

    # Adjust colorbar thickness
    #cbar.ax.set_aspect(20)  # Increase aspect ratio to make it thicker
   
    # Set labels and title
    ax1.set_xlabel("PC 1", fontsize = 20)
    ax1.set_ylabel("PC 2", fontsize = 20)
    ax1.tick_params(axis='both', which='major', labelsize=20)  # Increase tick sizes
    #ax1.legend(loc='upper right', fontsize= 24)
    ax1.set_title("")

    # Save or show the plot
    if output_file_with_snapshots:
        plt.savefig(output_file_with_snapshots, dpi=300, bbox_inches='tight')
        print(f"Static trajectory plot WITH snapshots saved to {output_file_with_snapshots}")
        plt.close(fig1)
    else:
        plt.show()

    # **Plot 2: Without Snapshots**
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    # Plot only source, intermediates, and target
    ax2.scatter(X1_vis[:, 0], X1_vis[:, 1], color=color_map[source_t], alpha=1.0, s=8,  zorder = 15, label=f'Time {source_t} (Training Data)')
    #ax2.scatter(Xm_vis[:, 0], Xm_vis[:, 1], color=color_map[middle_t], alpha=1.0, s=10,  zorder = 10, label=f'Time {middle_t}')
    ax2.scatter(X2_vis[:, 0], X2_vis[:, 1], color=color_map[target_t], alpha=1.0, s=8,  zorder = 10, label=f'Time {target_t} (Training Data)')

    # Set labels and title
    ax2.set_xlabel("PC 1", fontsize = 20)
    ax2.set_ylabel("PC 2", fontsize = 20)
    ax2.tick_params(axis='both', which='major', labelsize=20)  # Increase tick sizes
    #ax2.legend(loc='upper right', fontsize='small')
    ax2.set_title("")

    # Save or show the plot
    if output_file_without_snapshots:
        plt.savefig(output_file_without_snapshots, dpi=300, bbox_inches='tight')
        print(f"Static trajectory plot WITHOUT snapshots saved to {output_file_without_snapshots}")
        plt.close(fig2)
    else:
        plt.show()


## Static plot function for simple GPA (breast cancer cell line data)

def generate_static_trajectory_plots_two_timepoints_no_middle(pca,physical_dt,days, intermediate_days, X1_trpts, mats, d_red=26, output_file_with_snapshots=None, output_file_without_snapshots=None, output_file_snapshots_only=None):
    """
    Generate two static trajectory plots:
    1. With snapshots from X1_trpts using a color gradient.
    2. Without snapshots, showing only main time points.
    """
    

    # Define color gradient for snapshots
    num_snapshots = len(X1_trpts)
    colormap = cm.viridis  # Can change to "plasma", "inferno", etc.
    snapshot_colors = [colormap(i / num_snapshots) for i in range(num_snapshots)]

    # Rescale time values for the color bar
    time_values = np.linspace(0, physical_dt * num_snapshots, num_snapshots)

    # Create a normalization object for the color mapping
    norm = mcolors.Normalize(vmin=time_values.min(), vmax=time_values.max())
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])  # Needed for color bar

    source_t, middle_t, target_t = days[0], days[1], days[-1]
    
    # Define colors for time points
    color_map = {
        source_t: '#1f77b4',  # Blue
        #intermediate_days[0]: '#ff7f0e',  # Orange
        target_t: '#d62728'  # Red
    }

    # **Plot 1: With Snapshots**
    fig1, ax1 = plt.subplots(figsize=(8, 6))

    # Plot source, intermediates, and target
    X1_vis = pca.transform(mats[source_t])
    #Xm_vis = pca.transform(mats[middle_t])
    X2_vis = pca.transform(mats[target_t])
    #ax1.scatter(X1_vis[:, 0], X1_vis[:, 1], facecolors='none', edgecolors=color_map[source_t], linewidths=0.5, alpha=1.0, s=20, zorder=10, label=f'Time {source_t}')
    #ax1.scatter(X2_vis[:, 0], X2_vis[:, 1], facecolors='none', edgecolors=color_map[target_t], linewidths=0.5, alpha=1.0, s=20, zorder=10, label=f'Time {target_t}')



    # Plot snapshots from X1_trpts with a color gradient
    for i, X1_trpt in enumerate(X1_trpts):
        if np.isnan(X1_trpt).any():
            continue
        X1_hat_vis = X1_trpt
        ax1.scatter(X1_hat_vis[:, 0], X1_hat_vis[:, 1], color=snapshot_colors[i], alpha=0.75, s=2, zorder = 1)

    # Add a small color bar inside the plot
    cax = ax1.inset_axes([1.02, 0.2, 0.03, 0.6])  # [x, y, width, height] (relative position)
    
    # Create the colorbar with increased size
    cbar = plt.colorbar(sm, cax=cax)
    
    # Set manual tick positions
    cbar.set_ticks(np.linspace(0, 4, 5))  # Ensures ticks at 0, 1, 2, 3, 4
    
    # Optional: Explicitly set tick labels if needed
    cbar.set_ticklabels([0, 1, 2, 3, 4])  
    
    # Increase colorbar label font size
    cbar.set_label("Time", fontsize=20)  
    
    # Increase colorbar tick font size
    cbar.ax.tick_params(labelsize=20)

    # Adjust colorbar thickness
    #cbar.ax.set_aspect(20)  # Increase aspect ratio to make it thicker
   
    # Set labels and title
    ax1.set_xlabel("PC 1", fontsize = 20)
    ax1.set_ylabel("PC 2", fontsize = 20)
    ax1.tick_params(axis='both', which='major', labelsize=20)  # Increase tick sizes
    #ax1.legend(loc='upper right', fontsize= 24)
    ax1.set_title("")

    # Save or show the plot
    if output_file_with_snapshots:
        plt.savefig(output_file_with_snapshots, dpi=300, bbox_inches='tight')
        print(f"Static trajectory plot WITH snapshots saved to {output_file_with_snapshots}")
        plt.close(fig1)
    else:
        plt.show()

    # **Plot 2: Without Snapshots**
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    # Plot only source, intermediates, and target
    ax2.scatter(X1_vis[:, 0], X1_vis[:, 1], color=color_map[source_t], alpha=1.0, s=8,  zorder = 15, label=f'Time {source_t} (Training Data)')
    #ax2.scatter(Xm_vis[:, 0], Xm_vis[:, 1], color=color_map[middle_t], alpha=1.0, s=10,  zorder = 10, label=f'Time {middle_t}')
    ax2.scatter(X2_vis[:, 0], X2_vis[:, 1], color=color_map[target_t], alpha=1.0, s=8,  zorder = 10, label=f'Time {target_t} (Training Data)')

    # Set labels and title
    ax2.set_xlabel("PC 1", fontsize = 20)
    ax2.set_ylabel("PC 2", fontsize = 20)
    ax2.tick_params(axis='both', which='major', labelsize=20)  # Increase tick sizes
    #ax2.legend(loc='upper right', fontsize='small')
    ax2.set_title("")

    # Save or show the plot
    if output_file_without_snapshots:
        plt.savefig(output_file_without_snapshots, dpi=300, bbox_inches='tight')
        print(f"Static trajectory plot WITHOUT snapshots saved to {output_file_without_snapshots}")
        plt.close(fig2)
    else:
        plt.show()

                                                
def generate_static_trajectory_plots_cell_types(pca,days,cell_types_by_day,mats, output_file_cell_type_source=None, output_file_cell_type_target=None, output_file_cell_type_legend=None):
    """
    Generate two static trajectory plots:
    1. With snapshots from X1_trpts using a color gradient.
    2. Without snapshots, showing only main time points.
    """

    source_t, middle_t, target_t = days[0], days[1], days[-1]
    
    # Define colors for time points
    color_map = {
        source_t: '#1f77b4',  # Blue
        #intermediate_days[0]: '#ff7f0e',  # Orange
        target_t: '#d62728'  # Red
    }

    # **Plot 1: With Snapshots**

    # Same PCA transformation and cell type extraction as before
    X1_vis = pca.transform(mats[source_t])
    X2_vis = pca.transform(mats[target_t])
    cell_types_X1 = cell_types_by_day[source_t]
    cell_types_X2 = cell_types_by_day[target_t]
    unique_cell_types = np.unique(np.concatenate([cell_types_X1, cell_types_X2]))
    cell_type_palette = dict(zip(unique_cell_types, sns.color_palette("tab20", len(unique_cell_types))))
    
    # -----------------------
    # Plot 1: X1 colored, X2 gray (no legend)
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(X2_vis[:, 0], X2_vis[:, 1], color='lightgray', alpha=0.5, s=8)
    for cell_type in unique_cell_types:
        idx = cell_types_X1 == cell_type
        ax1.scatter(X1_vis[idx, 0], X1_vis[idx, 1], 
                    color=cell_type_palette[cell_type], s=8, alpha=1.0)
    ax1.set_xlabel("PC 1", fontsize=20)
    ax1.set_ylabel("PC 2", fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax1.set_title(f"Untreated Samples colored by Cell Type", fontsize=18)
    plt.tight_layout()
    if output_file_cell_type_source:
        plt.savefig(output_file_cell_type_source, dpi=300, bbox_inches='tight')
        plt.close(fig1)
    else:
        plt.show()
    
    # -----------------------
    # Plot 2: X2 colored, X1 gray (no legend)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(X1_vis[:, 0], X1_vis[:, 1], color='lightgray', alpha=0.5, s=8)
    for cell_type in unique_cell_types:
        idx = cell_types_X2 == cell_type
        ax2.scatter(X2_vis[idx, 0], X2_vis[idx, 1], 
                    color=cell_type_palette[cell_type], s=8, alpha=1.0)
    ax2.set_xlabel("PC 1", fontsize=20)
    ax2.set_ylabel("PC 2", fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    ax2.set_title(f"Treated Samples colored by Cell Type", fontsize=18)
    plt.tight_layout()
    if output_file_cell_type_target:
        plt.savefig(output_file_cell_type_target, dpi=300, bbox_inches='tight')
        plt.close(fig2)
    else:
        plt.show()

        

    # Use circle markers instead of patches for legend
    legend_elements = [
        mlines.Line2D(
            [], [], marker='o', color='w',
            markerfacecolor=cell_type_palette[cell_type],
            markersize=8, label=cell_type
        )
        for cell_type in unique_cell_types
    ]
    
    # Create circle markers for legend entries
    legend_elements = [
        mlines.Line2D(
            [], [], marker='o', color='w',
            markerfacecolor=cell_type_palette[cell_type],
            markersize=8, label=cell_type
        )
        for cell_type in unique_cell_types
    ]
    
    # Create figure and axis (just for the legend)
    fig_leg, ax_leg = plt.subplots()
    fig_leg.set_figwidth(8)  # Initial size; will be adjusted
    fig_leg.set_figheight(6)
    
    # Hide axes
    ax_leg.axis('off')
    
    # Add legend to axis (not directly to plt)
    legend = ax_leg.legend(
        handles=legend_elements,
        loc='center',
        frameon=True,
        fontsize=14,
        ncol=1,
        title='Cell Types',
        title_fontsize=14,
        borderpad=1
    )
    
    # Resize the figure to tightly fit the legend
    fig_leg.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig_leg.dpi_scale_trans.inverted())
    fig_leg.set_size_inches(bbox.width + 0.5, bbox.height + 0.5)  # Add a little padding
    
    # Save only, no display
    if output_file_cell_type_legend:
        plt.savefig(output_file_cell_type_legend, dpi=300, bbox_inches='tight')
        plt.close(fig_leg)
    
## Statitc Plots of Subtrajectories

def generate_static_cluster_plot_target(pca,
    source_t, target_t, X1_trpts, mats, optimal_k, start_i, index,p, reverse=False, intermediate_t=[1,2,3], 
    d_red=2, random_state=42, exp_memo='experiment', output_file = None
):
    """
    Generate a static plot of all snapshots from X1_trpts, colored by sub-trajectories.
    
    Parameters:
    - pca - a two dimensional pca
    - source_t (int): Source time step.
    - target_t (int): Target time step.
    - X1_trpts: list of cell positions over time generated by velocity field
    - Mats:  dictionary that groups gene expression data by timepoint.
    - optimal_k (int): Number of clusters.
    - start_i (int): Starting index for X1_trpts.
    - index (int): Step size for selecting snapshots.
    -p: velocity model parameters
    - reverse (bool): Whether to reverse trajectory direction.
    - intermediate_t (list): List of intermediate time points.
    - d_red (int): PCA dimension reduction.
    - random_state (int): Random seed for clustering.
    - exp_memo (str): Experiment identifier for file naming.
    - output_file: The output filepath for the graph
    """
    
    

    # Compute trajectory integration
    dt = p['numerical_ts'][-1] / 200
    
 
    # Perform clustering on last day's cell states
    last_day = mats[target_t]
    last_day_reduced = pca.transform(last_day).astype(np.float32)
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=random_state)
    kmeans.fit(last_day_reduced)
    last_day_labels = kmeans.labels_

    # Classify final predicted states
    X1_hat_last = X1_trpts[-1].astype(np.float32)
    X1_hat_labels = kmeans.predict(X1_hat_last)

    # Define colors for each cluster
    default_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    subgroup_colors_blue = {label: default_colors[i] for i, label in enumerate(np.unique(X1_hat_labels))}
    subgroup_colors_red = {label: default_colors[i] for i, label in enumerate(np.unique(last_day_labels))}

   
    

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot last day's clusters (target) in gray as actual data
    X2_vis = pca.transform(mats[target_t])
    ax.scatter(X2_vis[:, 0], X2_vis[:, 1], color='lightgray', alpha=0.7, s=10, zorder=10, label='Data')
        
    # Plot transported states (X1_hat) using assigned cluster colors (predicted sub-trajectories)
    for i, X1_trpt in enumerate(X1_trpts):
        if i % index == 0 and i >= start_i:
            if np.isnan(X1_trpt).any():
                continue
            X1_hat_vis = X1_trpt
    
            for label in np.unique(X1_hat_labels):
                idx = (X1_hat_labels == label)
    
                # Scatter Plot: Individual Points
                ax.scatter(X1_hat_vis[idx, 0], X1_hat_vis[idx, 1],
                           c=subgroup_colors_blue[label], alpha=0.75, s=3, zorder=1,
                           label=f'Predicted Subtrajectory {label+1}' if i == start_i else None)
    
                # **Line Plot: Connect Points from Previous Iteration**
                if i > start_i:  # Ensure there's a previous iteration
                    prev_X1_hat_vis = X1_trpts[i - index]  # Get previous iteration
                    prev_idx = (X1_hat_labels == label)
    
                    ax.plot([prev_X1_hat_vis[prev_idx, 0], X1_hat_vis[idx, 0]],
                            [prev_X1_hat_vis[prev_idx, 1], X1_hat_vis[idx, 1]],
                            color=subgroup_colors_blue[label], alpha=0.5, linewidth=1, zorder=0)

    
    # Plot source and intermediate days in gray circles (empty)
    X1_vis = pca.transform(mats[source_t])
    ax.scatter(X1_vis[:, 0], X1_vis[:, 1], color='lightgray', alpha=0.7, s=10, zorder=10)
    
    for t in intermediate_t:
        X_intermediate_vis = pca.transform(mats[t])
        ax.scatter(X_intermediate_vis[:, 0], X_intermediate_vis[:, 1],
                   color='lightgray', alpha=0.7, s=10, zorder=10)
    
    # Set labels and title
    ax.set_xlabel("PC 1", fontsize = 24)
    ax.set_ylabel("PC 2", fontsize = 24)
    ax.tick_params(axis='both', which='major', labelsize=24)  # Increases tick font size
    ax.set_title("")
    
    # Add custom legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    #ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='24')
    
    # Save or show the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Static cluster plot saved to {output_file}")
    plt.close(fig)


    return X1_hat_labels




def generate_static_cluster_plot_source(
    pca,
    source_t, target_t, X1_trpts, mats, optimal_k, start_i, index,p, reverse=False, intermediate_t=[1,2,3], 
    d_red=2, random_state=42, exp_memo='experiment', output_file = None
):
    """
    Generate a static plot of all snapshots from X1_trpts, colored by sub-trajectories.
    
    Parameters:
    - pca - a two dimensional pca
    - source_t (int): Source time step.
    - target_t (int): Target time step.
    - X1_trpts: list of cell positions over time generated by velocity field
    - Mats:  dictionary that groups gene expression data by timepoint.
    - optimal_k (int): Number of clusters.
    - start_i (int): Starting index for X1_trpts.
    - index (int): Step size for selecting snapshots.
    -p: velocity model parameters
    - reverse (bool): Whether to reverse trajectory direction.
    - intermediate_t (list): List of intermediate time points.
    - d_red (int): PCA dimension reduction.
    - random_state (int): Random seed for clustering.
    - exp_memo (str): Experiment identifier for file naming.
    - output_file: The output filepath for the graph
    """
    

    # Compute trajectory integration
    dt = p['numerical_ts'][-1] / 200
   
    # Perform clustering on last day's cell states
    last_day = mats[source_t]
    last_day_reduced = pca.transform(last_day).astype(np.float32)
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=random_state)
    kmeans.fit(last_day_reduced)
    last_day_labels = kmeans.labels_

    # Classify final predicted states
    X1_hat_last = X1_trpts[0].astype(np.float32)
    X1_hat_labels = kmeans.predict(X1_hat_last)

    # Define colors for each cluster
    default_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    subgroup_colors_blue = {label: default_colors[i] for i, label in enumerate(np.unique(X1_hat_labels))}
    subgroup_colors_red = {label: default_colors[i] for i, label in enumerate(np.unique(last_day_labels))}


    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot last day's clusters (target) in red subgroup colors
   # Plot last day's clusters (target) in gray as actual data
    X2_vis = pca.transform(mats[target_t])
    ax.scatter(X2_vis[:, 0], X2_vis[:, 1], facecolors='none', edgecolors='gray', linewidths=0.7, alpha=0.7, s=10, zorder=10, label='Data')


    # Plot transported states (X1_hat) using assigned cluster colors (predicted sub-trajectories)
    for i, X1_trpt in enumerate(X1_trpts):
        if i % index == 0 and i >= start_i:
            if np.isnan(X1_trpt).any():
                continue
            X1_hat_vis = X1_trpt
            for label in np.unique(X1_hat_labels):
                idx = (X1_hat_labels == label)
                ax.scatter(X1_hat_vis[idx, 0], X1_hat_vis[idx, 1],
                           c=subgroup_colors_blue[label], alpha=0.75, s=3, zorder=1,
                           label=f'Predicted Subtrajectory {label+1}' if i == start_i else None) 
                

    # Plot source day
    X1_vis = pca.transform(mats[source_t])
    ax.scatter(X1_vis[:, 0], X1_vis[:, 1], facecolors='none', edgecolors='gray', linewidths=0.7,  alpha=0.7, s=10, zorder = 10, label=f'Source: Day {source_t}')

    # Plot intermediate time points
    for t in intermediate_t:
        X_intermediate_vis = pca.transform(mats[t])
        ax.scatter(X_intermediate_vis[:, 0], X_intermediate_vis[:, 1], facecolors='none', edgecolors='gray', linewidths=0.7,  alpha=0.7, s=10, zorder = 10, label=f'Intermediate: Day {t}')

    # Set labels, legend, and title
    ax.set_xlabel("PC 1", fontsize = 24)
    ax.set_ylabel("PC 2", fontsize = 24)
    ax.tick_params(axis='both', which='major', labelsize=24)  # Increases tick font size
    ax.set_title("")

    # Save or show the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Static cluster plot saved to {output_file}")
    plt.close(fig)

    return X1_hat_labels





def classify_X1_hat(full_matrix,pca,source_t, target_t,X1_trpts,mats, optimal_k, start_i, index,p, reverse=True, intermediate_t=[1], 
    d_red=2, random_state=42, exp_memo='2', output_file = None,output_file_2 = None):
    

    dt = p['numerical_ts'][-1] / 200
    

    physical_dt = dt * p['ts'][-1] / p['numerical_ts'][-1]

    intermediate_t = np.array(intermediate_t)
    if len(intermediate_t) == 0:
        intermediate_t = range(source_t+1, target_t)

    day1, day2 = source_t, target_t

    # Perform clustering analysis on the last day's cell states
    last_day = mats[day2]
    last_day_reduced = pca.transform(last_day).astype(np.float32)

    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(last_day_reduced)
    last_day_labels = kmeans.labels_

    X1_hat_last = X1_trpts[-1].astype(np.float32)
    X1_hat_labels = kmeans.predict(X1_hat_last)

    # Generate colors for clusters
    default_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    viridis_colors = default_colors[:optimal_k]

    def get_subgroup_colors(labels, colors):
        unique_labels = np.unique(labels)
        subgroup_colors = {label: colors[i] for i, label in enumerate(unique_labels)}
        return subgroup_colors

    subgroup_colors_blue = get_subgroup_colors(X1_hat_labels, viridis_colors)
    subgroup_colors_red = get_subgroup_colors(last_day_labels, viridis_colors)

    # Define filename paths
    direction = 'backward' if reverse else 'forward'
    img_src = output_file
    initial_img_src = output_file_2
    # img_src = f"{output_dir}{exp_memo}-movie-cluster-{optimal_k}-{direction}-trajectory.gif"
    # initial_img_src = os.path.join(output_dir, f"{exp_memo}_initial_state_with_background.png")  # NEW STATIC FIGURE

    # Plot initial state for animation
    fig, ax = plt.subplots()
    ims = []

    # Prepare Data for Initial State
    reducer = decomposition.PCA(n_components=2, random_state=0)
    reducer.fit(full_matrix)
    vis_all_days = reducer.transform(full_matrix)
    
    X1_vis = reducer.transform(mats[day1])
    X2_vis = reducer.transform(mats[day2])

    # **(1) Save the Initial State Figure with Black Circle Outlines**
    fig_init, ax_init = plt.subplots(figsize=(8, 6))
    
    # Plot background gray cells
    ax_init.scatter(vis_all_days[:, 0], vis_all_days[:, 1], color='lightgray', alpha=1.0, s=8.0, zorder=5)
    
    # Plot last day's clusters **with black outline**
    scatter = ax_init.scatter(X2_vis[:, 0], X2_vis[:, 1], 
                              c=[subgroup_colors_red[label] for label in last_day_labels], 
                              alpha=1.0, s=50, edgecolors='black', linewidth=1.5, zorder=8)
    
    # Axis Labels
    ax_init.set_xlabel("PC 1", fontsize=24)
    ax_init.set_ylabel("PC 2", fontsize=24)
    ax_init.tick_params(axis='both', which='major', labelsize=24)
    ax_init.set_title("", fontsize=16)
    
    # Save the static figure
    plt.savefig(initial_img_src, dpi=300, bbox_inches="tight")
    plt.close()
    

    # **(2) Create a Separate Figure for the Legend**
    fig_legend, ax_legend = plt.subplots(figsize=(10, 2))  # Wider aspect ratio for horizontal layout
    ax_legend.axis("off")  # Hide axes
    
    # Get unique labels
    unique_labels = np.unique(last_day_labels)
    
    # Define legend elements:
    legend_elements = []
    
    # (A) **Fate Labels (Bold Dots with Black Outlines)**
    for i, label in enumerate(unique_labels):
        legend_elements.append(
            mlines.Line2D([], [], color=subgroup_colors_red[label], marker='o', linestyle='None', markersize=12, 
                          markeredgecolor='black', markeredgewidth=3.0, label=f"Fate {i+1}")
        )
    
    # (B) **Predicted Trajectories (One Dot with a Centered Horizontal Bar)**
    for i, label in enumerate(unique_labels):
        # Single dot with a horizontal bar
        trajectory_dot_bar = mlines.Line2D([], [], color=subgroup_colors_red[label], marker='o', linestyle='-', 
                                           markersize=6, linewidth=2, alpha=1.0, label=f"Trajectory {i+1}")
    
        # Add to legend
        legend_elements.append(trajectory_dot_bar)
    
    # Create horizontal legend **with a frame**
    ax_legend.legend(
        handles=legend_elements,
        loc="center", fontsize=20, title="Cell Fates & Predicted Trajectories",
        title_fontsize=20, ncol=4, frameon=True, framealpha=1.0, edgecolor="black", handletextpad=1.0, columnspacing=1.0
    )
    
    # Save the legend figure
    legend_img_src = initial_img_src.replace(".png", "_legend.png")
    plt.savefig(legend_img_src, dpi=300, bbox_inches="tight")
    plt.close()
    


    # Animation: Initial frame
    im = ax.scatter(X2_vis[:, 0], X2_vis[:, 1], 
                    c=[subgroup_colors_red[label] for label in last_day_labels], 
                    alpha=1.0, s=3.0, zorder=8)

    ttl = ax.text(0.5, 1.05, "t = %.3f" % (0), 
                  bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5}, 
                  transform=ax.transAxes, ha="center")

    ims.append([im, ttl])

    # Animation: Trajectory updates
    indices = range(len(X1_trpts) - start_i)
    if reverse:
        indices = reversed(indices)

    for i in indices:
        if i % index == 0:
            X1_trpt = X1_trpts[i]
            if np.isnan(X1_trpt).any():
                break
            X1_hat = pca.inverse_transform(X1_trpt)
            X1_hat_vis = reducer.transform(X1_hat)

            im = ax.scatter(X1_hat_vis[:, 0], X1_hat_vis[:, 1], 
                            c=[subgroup_colors_blue[label] for label in X1_hat_labels], 
                            alpha=1.0, s=3.0, zorder=10)
            
            ax.scatter(X2_vis[:, 0], X2_vis[:, 1], 
                       c=[subgroup_colors_red[label] for label in last_day_labels], 
                       alpha=1.0, s=3.0, zorder=8)

            # Keep background cells in the animation
            ax.scatter(vis_all_days[:, 0], vis_all_days[:, 1], color='lightgray', alpha=1.0, s=0.5, zorder=5)

            ttl = ax.text(0.5, 1.05, "t = %.3f" % (physical_dt * i), 
                          bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5}, 
                          transform=ax.transAxes, ha="center")

            ims.append([im, ttl])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=200)
    writergif = animation.PillowWriter(fps=3)
    ani.save(img_src, writer=writergif)
    plt.clf()
    
    # Display saved animation
    display(Image(filename=img_src))

    print(f"Initial state figure (with background) saved at: {initial_img_src}")
    print(f"Animation saved at: {img_src}")





def classify_X2_hat(
    full_matrix,pca,source_t, target_t,X1_trpts,mats, optimal_k, start_i, index,p, reverse=True, intermediate_t=[1], 
    d_red=2, random_state=42, exp_memo='2', output_file = None,output_file_2 = None):
    
   

    dt = p['numerical_ts'][-1] / 200
   
    physical_dt = dt * p['ts'][-1] / p['numerical_ts'][-1]

    intermediate_t = np.array(intermediate_t)
    if len(intermediate_t) == 0:
        intermediate_t = range(source_t+1, target_t)

    day1, day2 = source_t, target_t

    # Perform clustering analysis on the last day's cell states
    last_day = mats[day1]
    last_day_reduced = pca.transform(last_day).astype(np.float32)

    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(last_day_reduced)
    last_day_labels = kmeans.labels_

    X1_hat_last = X1_trpts[0].astype(np.float32)
    X1_hat_labels = kmeans.predict(X1_hat_last)

    # Generate colors for clusters
    default_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    viridis_colors = default_colors[:optimal_k]

    def get_subgroup_colors(labels, colors):
        unique_labels = np.unique(labels)
        subgroup_colors = {label: colors[i] for i, label in enumerate(unique_labels)}
        return subgroup_colors

    subgroup_colors_blue = get_subgroup_colors(X1_hat_labels, viridis_colors)
    subgroup_colors_red = get_subgroup_colors(last_day_labels, viridis_colors)

    # Define filename paths
    direction = 'backward' if reverse else 'forward'
    img_src = output_file
    initial_img_src = output_file_2
    
    # Plot initial state for animation
    fig, ax = plt.subplots()
    ims = []

    reducer = decomposition.PCA(n_components=2, random_state=0)
    reducer.fit(full_matrix)
    vis_all_days = reducer.transform(full_matrix)
    # Prepare Data for Initial State
    X1_vis = reducer.transform(mats[day1])
    X2_vis = reducer.transform(mats[day2])

    # **(1) Save the Initial State Figure with Black Circle Outlines**
    fig_init, ax_init = plt.subplots(figsize=(8, 6))
    
    # Plot background gray cells
    ax_init.scatter(vis_all_days[:, 0], vis_all_days[:, 1], color='lightgray', alpha=1.0, s=8.0, zorder=5)
    
    # Plot last day's clusters **with black outline**
    scatter = ax_init.scatter(X1_vis[:, 0], X1_vis[:, 1], 
                              c=[subgroup_colors_red[label] for label in last_day_labels], 
                              alpha=1.0, s=50, edgecolors='black', linewidth=1.5, zorder=8)
    
    # Axis Labels
    ax_init.set_xlabel("PC 1", fontsize=24)
    ax_init.set_ylabel("PC 2", fontsize=24)
    ax_init.tick_params(axis='both', which='major', labelsize=24)
    ax_init.set_title("", fontsize=16)
    
    # Save the static figure
    plt.savefig(initial_img_src, dpi=300, bbox_inches="tight")
    plt.close()
    

    # **(2) Create a Separate Figure for the Legend**
    fig_legend, ax_legend = plt.subplots(figsize=(10, 2))  # Wider aspect ratio for horizontal layout
    ax_legend.axis("off")  # Hide axes
    
    # Get unique labels
    unique_labels = np.unique(last_day_labels)
    
    # Define legend elements:
    legend_elements = []
    
    # (A) **Fate Labels (Bold Dots with Black Outlines)**
    for i, label in enumerate(unique_labels):
        legend_elements.append(
            mlines.Line2D([], [], color=subgroup_colors_red[label], marker='o', linestyle='None', markersize=12, 
                          markeredgecolor='black', markeredgewidth=3.0, label=f"Ancestor {i+1}")
        )
    
    # (B) **Predicted Trajectories (One Dot with a Centered Horizontal Bar)**
    for i, label in enumerate(unique_labels):
        # Single dot with a horizontal bar
        trajectory_dot_bar = mlines.Line2D([], [], color=subgroup_colors_red[label], marker='o', linestyle='-', 
                                           markersize=6, linewidth=2, alpha=1.0, label=f"Trajectory {i+1}")
    
        # Add to legend
        legend_elements.append(trajectory_dot_bar)
    
    # Create horizontal legend **with a frame**
    ax_legend.legend(
        handles=legend_elements,
        loc="center", fontsize=20, title="Cell Ancestors & Predicted Trajectories",
        title_fontsize=20, ncol=4, frameon=True, framealpha=1.0, edgecolor="black", handletextpad=1.0, columnspacing=1.0
    )
    
    # Save the legend figure
    legend_img_src = initial_img_src.replace(".png", "_legend.png")
    plt.savefig(legend_img_src, dpi=300, bbox_inches="tight")
    plt.close()
    


    # Animation: Initial frame
    im = ax.scatter(X1_vis[:, 0], X1_vis[:, 1], 
                    c=[subgroup_colors_red[label] for label in last_day_labels], 
                    alpha=1.0, s=3.0, zorder=8)

    ttl = ax.text(0.5, 1.05, "t = %.3f" % (0), 
                  bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5}, 
                  transform=ax.transAxes, ha="center")

    ims.append([im, ttl])

    # Animation: Trajectory updates
    indices = range(len(X1_trpts) - start_i)
    if reverse:
        indices = reversed(indices)

    for i in indices:
        if i % index == 0:
            X1_trpt = X1_trpts[i]
            if np.isnan(X1_trpt).any():
                break
            X1_hat = pca.inverse_transform(X1_trpt)
            X1_hat_vis = reducer.transform(X1_hat)

            im = ax.scatter(X1_hat_vis[:, 0], X1_hat_vis[:, 1], 
                            c=[subgroup_colors_blue[label] for label in X1_hat_labels], 
                            alpha=1.0, s=3.0, zorder=10)
            
            ax.scatter(X1_vis[:, 0], X1_vis[:, 1], 
                       c=[subgroup_colors_red[label] for label in last_day_labels], 
                       alpha=1.0, s=3.0, zorder=8)

            # Keep background cells in the animation
            ax.scatter(vis_all_days[:, 0], vis_all_days[:, 1], color='lightgray', alpha=1.0, s=0.5, zorder=5)

            ttl = ax.text(0.5, 1.05, "t = %.3f" % (physical_dt * i), 
                          bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5}, 
                          transform=ax.transAxes, ha="center")

            ims.append([im, ttl])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=200)
    writergif = animation.PillowWriter(fps=3)
    ani.save(img_src, writer=writergif)
    plt.clf()
    
    # Display saved animation
    display(Image(filename=img_src))

    print(f"Initial state figure (with background) saved at: {initial_img_src}")
    print(f"Animation saved at: {img_src}")

    










    







    




