import noise
import numpy as np
import matplotlib.pyplot as plt

def generate_perlin_noise(width, height, seed=42, movement_x=0, movement_y=0, scale=10.0):
    matrix = np.zeros((width, height, 2))

    for x in range(width):
        for y in range(height):
            value = noise.pnoise2((x + movement_x) / scale, (y + movement_y) / scale, octaves=6, persistence=0.5, lacunarity=2.0, repeatx=1024, repeaty=1024, base=seed)
            deg = value * 2 * np.pi
            matrix[x, y] = np.array([np.cos(deg), np.sin(deg)])

    # print(matrix)
    return matrix

if __name__ == "__main__":
    WIDTH = HEIGHT = 800  # pixels
    GRID_SIZE = 25  # pixels
    NUMBER_OF_POINTS = 100

    SCALE = 2
    SEED = np.random.randint(34)
    grid_matrix = generate_perlin_noise(WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE, SEED, SCALE)
    plt.imshow(grid_matrix[:,:,1], cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.show()


def get_line_pixels(x1, y1, x2, y2):
    pixels = []

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while (x1, y1) != (x2, y2):
        pixels.append((x1, y1))
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return pixels

def get_line_pixels_with_thickness(x1, y1, x2, y2, thickness):
    pixels = []

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while (x1, y1) != (x2, y2):
        # Add the original pixel
        pixels.append((x1, y1))

        # Add pixels in a neighborhood around the original pixel
        for i in range(1, thickness + 1):
            for j in range(1, thickness + 1):
                pixels.append((x1 + i, y1 + j))
                pixels.append((x1 + i, y1 - j))
                pixels.append((x1 - i, y1 + j))
                pixels.append((x1 - i, y1 - j))

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return pixels