# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import itertools # To cycle through colors for products
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # To draw filled faces
import matplotlib.patches as mpatches # For legend proxies

class BoxPlotter:
    """
    A class to visualize a container box and placed product boxes in 3D,
    including product weights.
    """

    def _plot_cube_edges(self, ax, xmin, xmax, ymin, ymax, zmin, zmax, **kwargs):
        """Helper method to plot the 12 edges of a cube."""
        # Faces parallel to XY plane (const Z)
        ax.plot([xmin, xmax], [ymin, ymin], zmin, **kwargs) # Bottom-Back Edge
        ax.plot([xmin, xmax], [ymax, ymax], zmin, **kwargs) # Top-Back Edge
        ax.plot([xmin, xmin], [ymin, ymax], zmin, **kwargs) # Left-Back Edge
        ax.plot([xmax, xmax], [ymin, ymax], zmin, **kwargs) # Right-Back Edge
        ax.plot([xmin, xmax], [ymin, ymin], zmax, **kwargs) # Bottom-Front Edge
        ax.plot([xmin, xmax], [ymax, ymax], zmax, **kwargs) # Top-Front Edge
        ax.plot([xmin, xmin], [ymin, ymax], zmax, **kwargs) # Left-Front Edge
        ax.plot([xmax, xmax], [ymin, ymax], zmax, **kwargs) # Right-Front Edge
        # Edges parallel to Z axis (const X, Y)
        ax.plot([xmin, xmin], [ymin, ymin], [zmin, zmax], **kwargs) # Back-Left Vertical Edge
        ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **kwargs) # Back-Right Vertical Edge
        ax.plot([xmin, xmin], [ymax, ymax], [zmin, zmax], **kwargs) # Top-Left Vertical Edge
        ax.plot([xmax, xmax], [ymax, ymax], [zmin, zmax], **kwargs) # Top-Right Vertical Edge


    def plot(self, data):
        """
        Plots the container and product placements based on the input data structure.

        Args:
            data (dict): A dictionary containing container and placement info.
                         Expected structure:
                         {
                             "container": {"name": str, "width": float, "height": float, "depth": float, ...},
                             "placements": [
                                 {"id": str, "x": float, "y": float, "z": float,
                                  "w": float, "h": float, "d": float, "weight": float, ...}, // Added weight
                                 ...
                             ]
                         }
                         *** Axis Mapping Change ***
                         Assumes container origin at (0,0,0).
                         Maps: width -> X, height -> Y, depth -> Z.
                         Placement (x,y,z) is the corner with min x (Width), min y (Height), min z (Depth).
                         Y-axis (Height) and Z-axis (Depth) are positive upwards/forwards.
        """
        container_info = data.get("container", {})
        placements_info = data.get("placements", [])

        if not container_info:
            print("Error: Container information missing in data.")
            return
        if not placements_info:
            print("Warning: No placement information found in data.")
            # Continue to plot empty container if desired, or return

        # --- Axis Mapping Change: Extract container dimensions ---
        box_dim = {
            'w': container_info.get('width', 10),  # X-Axis
            'd': container_info.get('height', 10), # Y-Axis (now holds height)
            'h': container_info.get('depth', 10)   # Z-Axis (now holds depth)
        }
        container_name = container_info.get('name', 'Container')

        # Create a figure with a 3D subplot
        fig = plt.figure(figsize=(8, 7)) # Slightly larger figure
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"Visualisierung: Container '{container_name}'") # Add title

        # --- Outer Box (Container) ---
        # Using w, d, h based on the new mapping
        xmin_outer, xmax_outer = 0, box_dim['w'] # X-axis (Width)
        ymin_outer, ymax_outer = 0, box_dim['d'] # Y-axis (Height)
        zmin_outer, zmax_outer = 0, box_dim['h'] # Z-axis (Depth)

        # Set axis limits based on the outer box dimensions
        # Add a small padding to prevent boxes touching the edge visually
        padding_x = box_dim['w'] * 0.05
        padding_y = box_dim['d'] * 0.05 # Padding based on height now
        padding_z = box_dim['h'] * 0.05 # Padding based on depth now
        ax.set(
            xlim=[xmin_outer - padding_x, xmax_outer + padding_x],
            ylim=[ymin_outer - padding_y, ymax_outer + padding_y], # Y limits based on height
            zlim=[zmin_outer - padding_z, zmax_outer + padding_z]  # Z limits based on depth
        )


        # Plot the edges of the outer box
        edges_kw_outer = dict(color='darkgrey', linewidth=1.5, linestyle='--', label='Container') # Changed style
        self._plot_cube_edges(ax, xmin_outer, xmax_outer, ymin_outer, ymax_outer, zmin_outer, zmax_outer, **edges_kw_outer)
        # Store handle for outer box legend entry
        outer_box_handle = plt.Line2D([0], [0], color=edges_kw_outer['color'], lw=edges_kw_outer['linewidth'], linestyle=edges_kw_outer['linestyle'], label=edges_kw_outer['label'])


        # --- Inner Product Boxes (Filled Faces) ---
        product_colors = itertools.cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']) # Tableau10 colors
        product_handles = [] # For legend

        for product in placements_info:
            # --- Axis Mapping Change: Extract product dimensions ---
            x_pos = product.get('x', 0)
            y_pos = product.get('y', 0) # Y-coordinate from data (interpreted as height pos)
            z_pos = product.get('z', 0) # Z-coordinate from data (interpreted as depth pos)
            w = product.get('w', 1)     # Width -> X-dim
            d = product.get('h', 1)     # Height -> Y-dim
            h = product.get('d', 1)     # Depth -> Z-dim
            product_id = product.get('id', 'Produkt') # Use ID from data
            weight = product.get('weight', 0.0) # --- Get weight ---

            # Calculate product box corner coordinates based on new mapping
            xmin_inner = x_pos
            xmax_inner = x_pos + w # X-span based on width
            ymin_inner = y_pos
            ymax_inner = y_pos + d # Y-span based on height
            zmin_inner = z_pos
            zmax_inner = z_pos + h # Z-span based on depth

            # Define the 8 vertices of the cube (coordinates map to X, Y, Z axes)
            v = np.array([
                [xmin_inner, ymin_inner, zmin_inner], [xmax_inner, ymin_inner, zmin_inner],
                [xmax_inner, ymax_inner, zmin_inner], [xmin_inner, ymax_inner, zmin_inner],
                [xmin_inner, ymin_inner, zmax_inner], [xmax_inner, ymin_inner, zmax_inner],
                [xmax_inner, ymax_inner, zmax_inner], [xmin_inner, ymax_inner, zmax_inner]
            ])

            # Define the 6 faces using vertex indices (topology remains the same)
            faces = [
                [v[0], v[1], v[5], v[4]], # Face parallel to XY @ zmin (Back face relative to Depth axis)
                [v[2], v[3], v[7], v[6]], # Face parallel to XY @ zmax (Front face relative to Depth axis)
                [v[0], v[3], v[7], v[4]], # Face parallel to XZ @ ymin (Bottom face relative to Height axis)
                [v[1], v[2], v[6], v[5]], # Face parallel to XZ @ ymax (Top face relative to Height axis)
                [v[0], v[1], v[2], v[3]], # Face parallel to YZ @ xmin (Left face relative to Width axis) - Reordered for consistency
                [v[4], v[5], v[6], v[7]], # Face parallel to YZ @ xmax (Right face relative to Width axis) - Reordered for consistency
            ]


            # Get color and create Poly3DCollection
            color = next(product_colors)
            collection = Poly3DCollection(faces, facecolors=color, linewidths=0.5, edgecolors='black', alpha=0.8) # Slightly increased alpha
            ax.add_collection3d(collection)

            # Create proxy artist for legend using the product ID
            product_handles.append(mpatches.Patch(color=color, label=f"{product_id} ({weight:.1f}g)", alpha=0.8)) # Add weight to legend label

            # --- Add Weight Text Label ---
            # Position text at the center of the top face (relative to height axis)
            text_x = xmin_inner + w / 2
            text_y = ymax_inner  # Y-coordinate of the top face
            text_z = zmin_inner + h / 2 # Z-coordinate (depth) center
            label_text = f"{weight:.1f}g" # Format weight string (assuming grams)
            ax.text(text_x, text_y, text_z, label_text,
                    ha='center', va='center', # Center text on the coordinate
                    color='black', # Text color
                    fontsize=7,    # Adjust font size as needed
                    zorder=1e4)    # Ensure text is drawn on top


        # --- Settings ---
        # --- Axis Mapping Change: Update Labels ---
        ax.set(
            xlabel='X (Breite) [cm]',
            ylabel='Y (Höhe) [cm]',   # Y-axis now represents Height
            zlabel='Z (Tiefe) [cm]',   # Z-axis now represents Depth
            # Adjust ticks based on actual limits including padding
            xticks=np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 5),
            yticks=np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 5),
            zticks=np.linspace(ax.get_zlim()[0], ax.get_zlim()[1], 5)
        )
        # Format ticks to fewer decimal places for cleaner look
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))


        ax.view_init(30, -60, 0) # Keep view angle
        # Adjust Zoom
        ax.set_box_aspect(None, zoom=0.85) # Slightly adjusted zoom

        # --- Legend ---
        # Combine outer box handle with product handles
        all_handles = [outer_box_handle] + product_handles
        if all_handles:
             # Place legend outside the plot area
             ax.legend(handles=all_handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8) # Adjust legend font size


        # Show the final plot
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
        plt.show()