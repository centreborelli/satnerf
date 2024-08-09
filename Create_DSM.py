import numpy as np
from PIL import Image
import open3d as o3d
from tqdm import tqdm

# Output paths: Set the paths for saving the results
output_ply_path = 'your_path/output.ply'  # Path for the point cloud PLY file
output_mesh_ply_path = 'your_path/mesh_output.ply'  # Path for the high-quality mesh PLY file
output_mesh_obj_path = 'your_path/mesh_output.obj'  # Path for the high-quality mesh OBJ file

# Display progress bar for DSM Image Load
print("Loading DSM Image...")
with tqdm(total=100, desc="Progress", ncols=100) as pbar:
    dsm_image = Image.open('your_tif_path/.tif')
    dsm_array = np.array(dsm_image)
    pbar.update(20)  # 20% completed

    

# Check if the DSM array is empty or not
if dsm_array.size == 0:
    raise ValueError("DSM image data is empty. Please check the input file.")

# Create Ply using vectorized numpy operations: Generate point cloud using vectorized operations
print("Creating Point Cloud...")
h, w = dsm_array.shape  # Get the height and width of the DSM image
x, y = np.meshgrid(np.arange(w), np.arange(h))  # Create a grid of x and y coordinates
z = dsm_array.flatten()  # Flatten the elevation data into a 1D array
points = np.vstack((x.flatten(), y.flatten(), z)).T  # Combine x, y, z coordinates into a single array to form points
pbar.update(20)  # 40% completed

# PointCloud Object Creation: Create a point cloud object using Open3D
print("Creating PointCloud Object...")
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)  # Set the coordinates in the point cloud
pbar.update(20)  # 60% completed

# **Normal Estimation**: Estimate normals for the point cloud
print("Estimating Normals...")
point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
# The radius and max_nn parameters define the search radius and the number of nearest neighbors to consider; adjust if needed
pbar.update(10)  # 70% completed

# Poisson Surface Reconstruction with GPU (if available): Generate a mesh using GPU
print("Performing Poisson Surface Reconstruction...")
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=12, scale=1.1)
# The depth and scale parameters control the level of detail and scaling of the mesh; adjust if needed
pbar.update(10)  # 80% completed

# Check if densities array is empty to avoid errors
densities_np = np.asarray(densities)
if densities_np.size == 0:
    raise ValueError("Densities array is empty. Poisson reconstruction might have failed. Check the input point cloud.")

# Remove low-density vertices to refine the mesh
print("Refining Mesh...")
vertices_to_remove = densities_np < np.quantile(densities_np, 0.01)  # Identify vertices to remove based on density threshold
if np.any(vertices_to_remove):
    mesh.remove_vertices_by_mask(vertices_to_remove)  # Remove the identified vertices from the mesh
pbar.update(10)  # 90% completed

# Save high-quality mesh to PLY and OBJ files
print("Saving Mesh and Point Cloud...")
o3d.io.write_triangle_mesh(output_mesh_ply_path, mesh)  # Save as PLY
o3d.io.write_triangle_mesh(output_mesh_obj_path, mesh)  # Save as OBJ
o3d.io.write_point_cloud(output_ply_path, point_cloud)  # Save the original point cloud to a PLY file
pbar.update(10)  # 100% completed

print("Process Completed.")
