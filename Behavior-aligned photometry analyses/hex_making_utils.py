def transform_scatter(points, image, rotate_center=(0,0)):
    fig, ax = plt.subplots(figsize=(12,10))
    plt.subplots_adjust(left=0.25, bottom=0.25)

    ax.imshow(image, extent=[0, 640, 0, 480])
    scatter = ax.scatter(points[:,0], points[:,1],marker='H',color='steelblue',alpha=0.5,s=1000)

    axcolor = 'lightgoldenrodyellow'
    ax_scale = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    ax_translate_x = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
    ax_translate_y = plt.axes([0.25, 0.0, 0.65, 0.03], facecolor=axcolor)
    ax_rotate = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

    s_scale = Slider(ax_scale, 'Scale', 0.1, 1.5, valinit=1.0)
    s_translate_x = Slider(ax_translate_x, 'Translate X', -200, 200, valinit=0)
    s_translate_y = Slider(ax_translate_y, 'Translate Y', -200, 200, valinit=0)
    s_rotate = Slider(ax_rotate, 'Rotate', -30, 30, valinit=0)

    def update(val):
        #clear_output()
        scale = s_scale.val
        translate_x = s_translate_x.val
        translate_y = s_translate_y.val
        rotate_deg = s_rotate.val

        # Scale
        scale_matrix = np.array([[scale, 0, 0],
                                 [0, scale, 0],
                                 [0, 0, 1]])

        # Translate
        translate_matrix = np.array([[1, 0, translate_x],
                                     [0, 1, translate_y],
                                     [0, 0, 1]])

        # Rotate
        rotate_rad = np.deg2rad(rotate_deg)
        rotate_matrix = np.array([[np.cos(rotate_rad), -np.sin(rotate_rad), 0],
                                  [np.sin(rotate_rad), np.cos(rotate_rad), 0],
                                  [0, 0, 1]])

        # Rotate around specified center
        center_matrix = np.array([[1, 0, rotate_center[0]],
                                  [0, 1, rotate_center[1]],
                                  [0, 0, 1]])
        center_translate_matrix = np.array([[1, 0, -rotate_center[0]],
                                            [0, 1, -rotate_center[1]],
                                            [0, 0, 1]])
        rotate_around_center_matrix = center_translate_matrix.dot(rotate_matrix).dot(center_matrix)

        # Transform points
        transform_matrix = rotate_around_center_matrix.dot(scale_matrix).dot(translate_matrix)
        transformed_points = transform_matrix.dot(points.T).T

        # Update scatter plot
        scatter.set_offsets(transformed_points[:,:2])
        #scatter.set_color(transformed_points[:,2])

        fig.canvas.draw_idle()
        global new_coords
        new_coords = transformed_points[:,:2]
        global transformParams
        print("rotate degree, scale, x tran, y tran = ",[rotate_deg,scale,translate_x,translate_y])
        transformParams = [rotate_deg,scale,translate_x,translate_y]
        #print(transformed_points)
    s_scale.on_changed(update)
    s_translate_x.on_changed(update)
    s_translate_y.on_changed(update)
    s_rotate.on_changed(update)

    plt.show()
    
def assign_to_hexagon(x, y, centroids):
    """
    Assigns each point in the given x-y coordinate vector to the closest hexagonal region of interest
    with the specified centroids.
    Parameters:
        x (np.ndarray): A 1D numpy array of x-coordinates.
        y (np.ndarray): A 1D numpy array of y-coordinates.
        centroids (np.ndarray): A 2D numpy array of hexagonal region centroids, where each row contains
            the x and y coordinates of the centroid of a region of interest.
    Returns:
        A 1D numpy array of integers, where each element is the index of the closest centroid in the
        centroids array.
    """
    # Calculate the Euclidean distance between each point and each centroid
    distances = np.sqrt((x[:, np.newaxis] - centroids[:, 0])**2 + (y[:, np.newaxis] - centroids[:, 1])**2)
    # Find the index of the closest centroid for each point
    closest_centroid_indices = np.argmin(distances, axis=1) +1
    
    return closest_centroid_indices

#compute the centroids of the initial hexagons
hexlist = [2,47,46,45,44,43,3,\
49,42,41,40,39,48,\
38,37,36,35,34,33,\
32,31,30,29,28,\
27,26,25,24,23,\
22,21,20,19,\
18,17,16,15,\
14,13,12,\
11,10,9,\
8,7,\
6,5,\
4,\
1]
#hexlist = np.subtract(hexlist,1) #convert to index-based states
coords = []
cols = [7,6,6,5,5,4,4,3,3,2,2,1,1]
maxrows = 13
r = 0
x = 1
y = 1
startr = 1
while r < maxrows:
    maxcols = cols[r]
    c = 0
    if r%2!=0:
        startr+=1
    x=startr
    while c < maxcols:
        coords.append([x,y])
        x += 2
        c += 1
    if r%2!=0:
        y += 2
    else:
        y+=1
    r += 1
cents = {h: c for h,c in zip(hexlist,coords)}


# define the centroids of the hexagons
centroids = np.zeros((49,2))
for h in list(cents.keys()):
    centroids[h-1] = cents[h]

centroids[:,0] = centroids[:,0]/centroids[:,0].max()*450 +80
centroids[:,1] = centroids[:,1]/centroids[:,1].max()*410

