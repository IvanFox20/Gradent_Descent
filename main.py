import numpy as np
import os
import random
import warnings

A = np.array(
    [[1.3, 0.4, 0.5],
     [0.4, 1.3, 0.3],
     [0.5, 0.3, 1.3]
     ], dtype=np.float32
)

B = np.array([0.25,
              0.51,
              1.44],
dtype=np.float32
)
def is_positive_definite(matrix):
    # Проверка на квадратную матрицу
    if matrix.shape[0] != matrix.shape[1]:
        return False

    n = matrix.shape[0]

    # Проверка положительности всех главных миноров
    for i in range(1, n + 1):
        submatrix = matrix[:i, :i]
        if np.linalg.det(submatrix) <= 0:
            return False

    return True


def is_symmetric(matrix):
    return (matrix == matrix.T).all()

def gradient_descent(A, B, iters=2000, eps=1e-4):
    # Initialize the variable vector x with zeros
    x = np.zeros_like(B, dtype=np.float64)

    for iter in range(iters):
        # Calculate the predicted values
        predictions = np.dot(A, x)

        # Calculate the error
        error = predictions - B

        R = B - np.dot(A, x)

        lambd = np.dot(R, R) / (np.dot(np.dot(A, R), R))

        # Calculate the gradient
        gradient = 2 * R #np.dot(A.T, error)

        # Update the variable vector using the gradient
        x += lambd * R
        print(x)
        # Check for convergence
        if np.linalg.norm(gradient) < eps:
            print(f"Converged in {iter+1} epochs.")
            break
    return x

if not (is_symmetric(A) and is_positive_definite(A)):
  A_new = np.dot(A.T, A)
  B_new = np.dot(A.T, B)
  solution = gradient_descent(A_new, B_new, iters=5000, eps=1e-10)
  print(solution, np.linalg.solve(A_new, B_new))
else:
  solution = gradient_descent(A, B)
  print("Solution:", solution)
  print(np.linalg.solve(A, B))