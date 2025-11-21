# --- Visualization tools ---
import matplotlib.pyplot as plt  # Plotting library
from mpl_toolkits.mplot3d import Axes3D  # Enables 3D plotting with matplotlib
from sklearn.manifold import TSNE        # Dimensionality reduction for visualization
import numpy as np

def tsne_plot(data):
    """
    Perform a 3D t-SNE dimensionality reduction and visualize the results.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        High-dimensional data to be visualized using t-SNE. Each row should
        represent one sample, and each column a feature.

    Notes
    -----
    - The function reduces the input data to 3 dimensions using t-SNE and plots
      it as a 3D scatter plot.
    - Each point is given a unique color based on its index.
    - The perplexity is set to `data.shape[0] - 1` (which may cause errors if
      you have very few data points or many; typically values between 5–50
      work best).
    """

    # Apply t-SNE to project the high-dimensional data down to 3D
    tsne = TSNE(
        n_components=3,
        random_state=42,
        perplexity=min(data.shape[0] - 1, 30)  # safer perplexity
    )
    data_3d = tsne.fit_transform(data)

    # Plotting: Create a 3D figure for visualization
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Assign a unique color to each point based on its index
    num_points = len(data_3d)
    colors = plt.cm.tab20(np.linspace(0, 1, num_points))

    # Scatter plot each point with its corresponding color
    for idx, point in enumerate(data_3d):
        ax.scatter(point[0], point[1], point[2], label=str(idx), color=colors[idx])

    # Label the axes and set the title
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    plt.title('3D t-SNE Visualization')

    # Show a legend mapping points to their indices
    plt.legend(title='Input Order', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()