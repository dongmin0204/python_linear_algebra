import numpy as np

class InvertibleMatrix:
    def __init__(self, size=4, value_range=(-10.0, 10.0)):
        self.size = size
        self.value_range = value_range
        self.matrix = self.generate_invertible_matrix()

    def generate_invertible_matrix(self):
        while True:
            # Generate a random matrix of the specified size with values within the given range
            matrix = np.random.randint(self.value_range[0], self.value_range[1] + 1, size=(self.size, self.size))
            # Check if the matrix is invertible by calculating its determinant
            if np.linalg.det(matrix) != 0:
                return matrix

    def display_matrix(self):
        print(self.matrix)

# Create an instance of the InvertibleMatrix class
for i in range(5):
    matrix_instance = InvertibleMatrix()
    matrix_instance.display_matrix()
