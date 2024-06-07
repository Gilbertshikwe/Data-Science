# The derivative measures how a function changes as its input changes.

# Import sympy library
import sympy as sp  #type:ignore # Correctly import sympy with alias 'sp'

# Symbolic variables
x = sp.Symbol('x')
y = sp.Symbol('y')

# Define a function
f = x**2 + 2*x*y
print("Function:", f)  # Output: Function: x**2 + 2*x*y

# Calculate derivatives
df_dx = sp.diff(f, x)
df_dy = sp.diff(f, y)
print("Derivative w.r.t x:", df_dx)  # Output: Derivative w.r.t x: 2*x + 2*y
print("Derivative w.r.t y:", df_dy)  # Output: Derivative w.r.t y: 2*x

# Partial Derivatives

# Define a new function with multiple variables
f = x**2 + y**2

# Calculate partial derivatives
f_partial_x = sp.diff(f, x)
f_partial_y = sp.diff(f, y)
print("Partial derivative with respect to x:", f_partial_x)  # Output: Partial derivative with respect to x: 2*x
print("Partial derivative with respect to y:", f_partial_y)  # Output: Partial derivative with respect to y: 2*y

