import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from spa import SuccessiveProj
from vertexH import plot_simplex
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go
import plotly.express as px

def get_projected_points(samples,K=3,return_X=False, low_dim=False):
    '''Input: samples = d * N_samples array
    Output: d * N_samples array where i^th column is the i^th projected point'''
    row_means = np.mean(samples, axis=1)
    means_expanded = np.outer(row_means, np.ones(samples.shape[1]))
    X_bar = samples - means_expanded    

    """SVD STEP"""
    U, _, _ = svd(X_bar)
    U_s = U[:,:K-1]

    if low_dim:
        return np.dot(np.transpose(U_s),samples)
    H = np.dot(U_s,np.transpose(U_s))

    if return_X:
        return means_expanded + np.dot(H,X_bar), X_bar, means_expanded
    return means_expanded + np.dot(H,X_bar)



# if __name__ == '__main__':

#     samples = np.load("RongChen/mat_5Y_last_new_data.npy").T
#     # Read the column names from the text file
#     with open("names.txt", "r") as file:
#         column_names = np.array(file.read().split("\n"))
#     samples = np.reshape(samples,(4,24))
#     proj = get_projected_points(samples)
#     vertices , id_country = SuccessiveProj(np.transpose(proj),K=4)
#     id_country = id_country.astype(np.int64)
#     print(id_country)
#     print(column_names)
#     vertex_label = column_names[id_country]

#     print(vertices.shape)
#     x, y, z = proj[0,:], proj[1,:],proj[2,:]

#     # Define the faces of the tetrahedron using the vertex indices
#     faces = [
#         [vertices[0], vertices[1], vertices[2]],
#         [vertices[0], vertices[1], vertices[3]],
#         [vertices[0], vertices[2], vertices[3]],
#         [vertices[1], vertices[2], vertices[3]]
#     ]


    # # Create a scatter3d plot for the vertices and scatter points
    # x = np.concatenate((vertices[:, 0], samples[0]))  # Combine tetrahedron vertices and point cloud
    # y = np.concatenate((vertices[:, 1], samples[1]))
    # z = np.concatenate((vertices[:, 2], samples[2]))

    # # Create labels for the vertices
    # #labels = ['A', 'B', 'C', 'D'] + ['Point'] * len(samples[0])

    # # Create a color array for distinguishing between vertices and scatter points
    # colors = ['blue'] * 4 + ['red'] * len(samples[0])

    # # Create a figure with scatter3d
    # fig = px.scatter_3d(
    #     x=x,
    #     y=y,
    #     z=z,
    #     text=labels,
    #     color=colors,
    #     size_max=5  # Adjust the size as needed
    # )

    # # Set axis labels and title
    # fig.update_layout(
    #     scene=dict(
    #         xaxis_title='X',
    #         yaxis_title='Y',
    #         zaxis_title='Z',
    #     ),
    #     title='Interactive 3D Plot with Tetrahedron, Scatter Points, and Labeled Vertices',
    # )

    # # Save the plot as an HTML file
    # fig.write_html("interactive_plot_with_tetrahedron_points_and_vertices.html")


    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # Create a Poly3DCollection to represent the tetrahedron
    # poly3d = [[*face] for face in faces]
    # tetrahedron = [[*face] for face in poly3d]
    # ax.add_collection3d(Poly3DCollection(tetrahedron, facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.2))
    # ax.scatter(x, y, z, c='red', marker='o', label="sample points")
    # for i, label in enumerate(vertex_label):
    #     ax.text(vertices[i,0] + 0.1, vertices[i,1] + 0.1, vertices[i,2] + 0.1, label, fontsize=12)

    

    # # Adjust axis limits to zoom in on the tetrahedron
    # ax.set_xlim(np.min(x)-0.05, np.max(x)+0.05)
    # ax.set_ylim(np.min(y)-0.05, np.max(y)+0.05)
    # ax.set_zlim(np.min(z)-0.05, np.max(z)+0.05)
    # # Create the scatter plot
    # # Set axis labels and limits
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # # Add a title
    # ax.set_title('Tetrahedron Contours')
    # plt.legend()

    # # Display the plot
    # plt.show()