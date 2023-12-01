from helpers import generate_perlin_noise, get_line_pixels_with_thickness, get_line_pixels
import numpy as np
import cv2

WIDTH , HEIGHT = 1080, 720 # pixels
GRID_SIZE = 5  # pixels
NUMBER_OF_POINTS = 1000

DRAW_VECTORS = False
DRAW_GRIDS = False
DRAW_POINTS = False
CHANGE_GRID = False

SPEED = 1

SCALE = 150
SEED = np.random.randint(34)
# SEED = 10

class Point:
    def __init__(self):
        if np.random.randint(100) > 50 :
            self.x = 0
            self.y = np.random.randint(HEIGHT)
        else:
            self.x = np.random.randint(HEIGHT)
            self.y = 0

        self.speed = SPEED
        self.velocity = np.zeros(2)
    def move(self, grid):
        last_x, last_y = self.x, self.y

        try:
            velocity_x, velocity_y = grid[self.x // GRID_SIZE, self.y // GRID_SIZE] * SEED
        except IndexError:
            velocity_x = velocity_y = 0
        
        self.x = int(self.x + velocity_x)
        self.y = int(self.y + velocity_y)

        return last_x, last_y

main_layer = np.zeros((HEIGHT, WIDTH, 3))
grid_layer = np.zeros((HEIGHT, WIDTH, 3))
vector_layer = np.zeros((HEIGHT, WIDTH, 3))

if DRAW_GRIDS:
    # Draw vertical lines
    grid_layer[::GRID_SIZE, :, 0] = 255
    # Draw horizontal lines
    grid_layer[:, ::GRID_SIZE, 0] = 255

def draw_vectors(layer, matrix):
    # Get the coordinates of the grid points
    # print(np.arange(0, WIDTH -24, GRID_SIZE))
    grid_points_x, grid_points_y = np.meshgrid(np.arange(0, WIDTH, GRID_SIZE), np.arange(0, HEIGHT, GRID_SIZE), indexing='ij')
    grid_points = np.stack((grid_points_x, grid_points_y), axis=-1)

    # Calculate end points for vectors
    # x = np.shape(matrix)[0]
    # y = np.shape(matrix)[1]
    # print(x)
    end_points = grid_points + matrix[:, :] * 10

    # Convert coordinates to integers
    grid_points = grid_points.astype(int)
    end_points = end_points.astype(int)

    print(end_points)

    # Reshape the arrays for cv2.arrowedLine
    grid_points = grid_points.reshape(-1, 2)
    end_points = end_points.reshape(-1, 2)

    # Draw arrows
    for start, end in zip(grid_points, end_points):
        cv2.arrowedLine(layer, tuple(start + GRID_SIZE//2), tuple(end + GRID_SIZE//2), (0, 0, 255), 1)
    
    return layer

points = [Point() for _ in range(NUMBER_OF_POINTS)]

# Animation loop
running = True
frame = 0
grid_matrix = generate_perlin_noise((WIDTH // GRID_SIZE) + 1, (HEIGHT // GRID_SIZE) + 1, SEED, frame, frame, SCALE)
while running:
    frame += 0.1
    key = cv2.waitKey(10)
    if key == 27:  # Press 'Esc' to exit
        running = False

    if CHANGE_GRID:
        grid_matrix = generate_perlin_noise((WIDTH // GRID_SIZE) + 1, (HEIGHT // GRID_SIZE) + 1, SEED, frame, frame, SCALE)

    for point in points:
        x, y = point.move(grid_matrix)
        try:
            # main_layer[point.y, point.x] = 255
            # cv2.line(main_layer, (x, y), (point.x, point.y), (1,1,1), thickness=2)
            path = get_line_pixels(x, y, point.x, point.y)
            # path = get_line_pixels_with_thickness(x, y, point.x, point.y, 2)
            for x, y in path:
                main_layer[y, x] += 0.1
        except IndexError:
            # points.remove(point)
            pass
    if DRAW_VECTORS:
        vector_layer.fill(0)
        vector_layer = draw_vectors(vector_layer, grid_matrix)

    result = cv2.addWeighted(grid_layer, 1, main_layer, 1, 0)
    result = cv2.addWeighted(vector_layer, 1, result, 1, 0)
    result[:,:,0] = 0
    # result[:,:,1] = 0
    result[:,:,2] = 0

    cv2.imshow('Flow Fields', result)

cv2.destroyAllWindows()
