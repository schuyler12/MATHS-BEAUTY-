import numpy as np
import matplotlib.pyplot as plt

def julia_set(c, num_points=1000, max_iter=100, threshold=2.0, output_file=None):
    """
    Generates and plots the Julia set for a given complex number c.
    
    Args:
        c (complex): The constant parameter for the quadratic map z = z^2 + c.
        num_points (int): The number of points along each axis (total points = num_points^2).
        max_iter (int): The maximum number of iterations.
        threshold (float): The escape threshold radius.
        output_file (str, optional): If provided, save the plot to this file instead of showing it.
    """
    # 1. Define the grid of points (initial z values)
    x = np.linspace(-2, 2, num_points)
    y = np.linspace(-2, 2, num_points)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    # Flatten the array to treat them as a list of points
    points = Z.flatten()
    
    # 2. Iterate the function z = z^2 + c
    # We keep track of which points have escaped
    # To correctly filter based on the definition "remove dead points at last",
    # we need to track if the orbit remains bounded throughout all iterations.
    
    # Initialize with all points valid
    # We will compute the orbit. If at any step |z| > threshold, it's "dead".
    
    current_z = points.copy()
    still_alive = np.ones(points.shape, dtype=bool)
    
    print(f"Iterating for {len(points)} points...")
    
    for i in range(max_iter):
        # Update only alive points to save computation
        # (Though numpy vectorization on full array is often faster than boolean indexing if array is small,
        # but for large arrays, masking is better)
        
        # Calculate next iteration for all (or just alive)
        # masking complicates in-place updates slightly if we want to keep index alignment,
        # but simpliest is to update everything or use 'where'
        
        mask_indices = np.where(still_alive)[0]
        if len(mask_indices) == 0:
            break
            
        current_z[mask_indices] = current_z[mask_indices] ** 2 + c
        
        # Check escape
        escaped = np.abs(current_z[mask_indices]) > threshold
        
        # Mark dead
        # indices of escaped points within the mask_indices array
        escaped_real_indices = mask_indices[escaped]
        still_alive[escaped_real_indices] = False
            
    # 3. Filter: Remove dead points, keep valid points
    # We want the INITIAL positions of the points that survived.
    valid_points_start = points[still_alive]
    
    print(f"Finished iterations. Found {len(valid_points_start)} valid points.")
    
    # 4. Plot
    plt.figure(figsize=(10, 10))
    plt.title(f"Julia Set for c = {c}")
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    
    if len(valid_points_start) > 0:
        plt.scatter(valid_points_start.real, valid_points_start.imag, s=1, c='black', marker='.')
    else:
        print("No points remained bounded.")

    plt.axis('equal')
    plt.grid(True)
    
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

if __name__ == "__main__":
 while True:
    print("Julia Set Visualization")
    try:
        r = float(input("Enter real part of c (e.g., -0.8): "))
        i = float(input("Enter imaginary part of c (e.g., 0.156): "))
        c_val = complex(r, i)
        
        resolution = input("Enter resolution (points per axis, default 500): ")
        if resolution.strip() == "":
            res = 500
        else:
            res = int(resolution)
            
        julia_set(c_val, num_points=res)
        
    except ValueError:
        print("Invalid input. Please enter numeric values.")
