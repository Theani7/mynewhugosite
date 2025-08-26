+++
title = "Implementing Linear Regression from Scratch: A Complete Guide"
date = 2025-08-20
tags = ["linear regression", "AI"]
+++

*Published on August 20, 2025*

Linear regression is the foundation of machine learning and statistics. While libraries like scikit-learn make it easy to apply, understanding how to implement it from scratch provides invaluable insights into the underlying mathematics and helps build intuition for more complex algorithms. In this comprehensive guide, we'll build linear regression step by step, covering both the theory and practical implementation.

## Understanding the Mathematics Behind Linear Regression

Linear regression aims to find the best-fitting line through a set of data points. For simple linear regression with one feature, we're looking for a line defined by:

```
y = mx + b
```

In machine learning notation, this becomes:

```
ŷ = θ₀ + θ₁x
```

Where:
- `ŷ` is our prediction
- `θ₀` is the intercept (bias term)
- `θ₁` is the slope (weight)
- `x` is our input feature

For multiple features, this extends to:

```
ŷ = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
```

## The Cost Function: Mean Squared Error

To find the best parameters, we need to define what "best" means. We use the Mean Squared Error (MSE) as our cost function:

```
J(θ) = (1/2m) Σ(ŷᵢ - yᵢ)²
```

Where:
- `m` is the number of training examples
- `ŷᵢ` is our prediction for example i
- `yᵢ` is the actual value for example i

The goal is to find the parameters θ that minimize this cost function.

## Implementation Approach: Gradient Descent

We'll use gradient descent to minimize our cost function. The algorithm works by:

1. Initialize parameters randomly
2. Calculate predictions using current parameters
3. Compute the cost
4. Calculate gradients (partial derivatives)
5. Update parameters in the direction that reduces cost
6. Repeat until convergence

## Complete Implementation from Scratch

Let's implement linear regression using only NumPy:

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

class LinearRegressionFromScratch:
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000, 
                 tolerance: float = 1e-8):
        """
        Initialize Linear Regression model
        
        Args:
            learning_rate: Step size for gradient descent
            max_iterations: Maximum number of training iterations
            tolerance: Convergence threshold
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.n_features = None
        
    def _add_bias_term(self, X: np.ndarray) -> np.ndarray:
        """Add bias term (column of ones) to feature matrix"""
        return np.column_stack([np.ones(X.shape[0]), X])
    
    def _compute_cost(self, X: np.ndarray, y: np.ndarray, 
                      theta: np.ndarray) -> float:
        """
        Compute Mean Squared Error cost
        
        Args:
            X: Feature matrix with bias term
            y: Target values
            theta: Parameters (bias + weights)
            
        Returns:
            Cost value
        """
        m = X.shape[0]
        predictions = X.dot(theta)
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        return cost
    
    def _compute_gradients(self, X: np.ndarray, y: np.ndarray, 
                           theta: np.ndarray) -> np.ndarray:
        """
        Compute gradients for gradient descent
        
        Args:
            X: Feature matrix with bias term
            y: Target values
            theta: Current parameters
            
        Returns:
            Gradients for each parameter
        """
        m = X.shape[0]
        predictions = X.dot(theta)
        errors = predictions - y
        gradients = (1 / m) * X.T.dot(errors)
        return gradients
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionFromScratch':
        """
        Train the linear regression model using gradient descent
        
        Args:
            X: Training features (m x n matrix)
            y: Training targets (m x 1 vector)
            
        Returns:
            Fitted model instance
        """
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        m, n = X.shape
        self.n_features = n
        
        # Add bias term
        X_with_bias = self._add_bias_term(X)
        
        # Initialize parameters
        np.random.seed(42)  # For reproducibility
        theta = np.random.normal(0, 0.01, X_with_bias.shape[1])
        
        # Gradient descent
        for iteration in range(self.max_iterations):
            # Compute cost
            cost = self._compute_cost(X_with_bias, y, theta)
            self.cost_history.append(cost)
            
            # Compute gradients
            gradients = self._compute_gradients(X_with_bias, y, theta)
            
            # Update parameters
            new_theta = theta - self.learning_rate * gradients
            
            # Check for convergence
            if np.allclose(theta, new_theta, atol=self.tolerance):
                print(f"Converged after {iteration + 1} iterations")
                break
                
            theta = new_theta
        
        # Store final parameters
        self.bias = theta[0]
        self.weights = theta[1:]
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if self.weights is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # Make predictions
        predictions = self.bias + X.dot(self.weights)
        return predictions
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R-squared score
        
        Args:
            X: Features
            y: True values
            
        Returns:
            R-squared score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2
    
    def plot_cost_history(self) -> None:
        """Plot the cost function over iterations"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title('Cost Function Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Cost (MSE)')
        plt.grid(True)
        plt.show()

# Analytical Solution (Normal Equation)
class LinearRegressionAnalytical:
    """Linear Regression using the analytical solution (Normal Equation)"""
    
    def __init__(self):
        self.weights = None
        self.bias = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionAnalytical':
        """
        Fit using normal equation: θ = (X^T X)^(-1) X^T y
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # Add bias term
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Normal equation
        theta = np.linalg.pinv(X_with_bias.T.dot(X_with_bias)).dot(X_with_bias.T).dot(y)
        
        self.bias = theta[0]
        self.weights = theta[1:]
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.bias + X.dot(self.weights)

# Example Usage and Demonstration
def generate_sample_data(n_samples: int = 100, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sample data for demonstration"""
    np.random.seed(42)
    X = np.random.uniform(-3, 3, n_samples)
    true_slope = 2.5
    true_intercept = 1.0
    y = true_intercept + true_slope * X + np.random.normal(0, noise, n_samples)
    return X, y

def compare_implementations():
    """Compare our implementation with analytical solution"""
    # Generate sample data
    X, y = generate_sample_data(n_samples=100, noise=0.3)
    
    # Gradient Descent Implementation
    model_gd = LinearRegressionFromScratch(learning_rate=0.01, max_iterations=1000)
    model_gd.fit(X, y)
    
    # Analytical Implementation
    model_analytical = LinearRegressionAnalytical()
    model_analytical.fit(X, y)
    
    # Make predictions
    X_test = np.linspace(-3, 3, 100)
    y_pred_gd = model_gd.predict(X_test)
    y_pred_analytical = model_analytical.predict(X_test)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Data and predictions
    plt.subplot(1, 3, 1)
    plt.scatter(X, y, alpha=0.6, color='blue', label='Data')
    plt.plot(X_test, y_pred_gd, 'r-', label='Gradient Descent', linewidth=2)
    plt.plot(X_test, y_pred_analytical, 'g--', label='Analytical', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Cost history
    plt.subplot(1, 3, 2)
    plt.plot(model_gd.cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Function Convergence')
    plt.grid(True)
    
    # Plot 3: Parameter comparison
    plt.subplot(1, 3, 3)
    methods = ['Gradient Descent', 'Analytical']
    biases = [model_gd.bias, model_analytical.bias]
    weights = [model_gd.weights[0], model_analytical.weights[0]]
    
    x = np.arange(len(methods))
    width = 0.35
    
    plt.bar(x - width/2, biases, width, label='Bias', alpha=0.8)
    plt.bar(x + width/2, weights, width, label='Weight', alpha=0.8)
    plt.xlabel('Method')
    plt.ylabel('Parameter Value')
    plt.title('Parameter Comparison')
    plt.xticks(x, methods)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison
    print("Parameter Comparison:")
    print(f"Gradient Descent - Bias: {model_gd.bias:.4f}, Weight: {model_gd.weights[0]:.4f}")
    print(f"Analytical - Bias: {model_analytical.bias:.4f}, Weight: {model_analytical.weights[0]:.4f}")
    print(f"R² Score (GD): {model_gd.score(X, y):.4f}")
    print(f"R² Score (Analytical): {model_analytical.score(X, y):.4f}")

if __name__ == "__main__":
    compare_implementations()
```

## Key Implementation Details

**Feature Scaling**: For real-world applications, consider implementing feature scaling to improve gradient descent convergence:

```python
def standardize_features(X):
    """Standardize features to have zero mean and unit variance"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std, mean, std
```

**Learning Rate Selection**: The learning rate is crucial for convergence. Too high and the algorithm might overshoot; too low and it converges slowly. Consider implementing adaptive learning rates or learning rate scheduling.

**Regularization**: For preventing overfitting, you can add L1 (Lasso) or L2 (Ridge) regularization terms to the cost function.

## Advantages and Limitations

**Advantages of our implementation**:
- Complete understanding of the underlying mathematics
- Full control over the optimization process
- Easy to extend with regularization or other modifications
- Educational value for understanding machine learning fundamentals

**Limitations compared to optimized libraries**:
- Slower execution for large datasets
- Less numerical stability for ill-conditioned problems
- Missing advanced features like automatic differentiation
- No built-in cross-validation or model selection tools

## When to Use Each Approach

**Gradient Descent** is preferred when:
- Working with large datasets where matrix inversion is computationally expensive
- The feature matrix is not invertible
- You want to understand the optimization process
- Implementing online learning scenarios

**Analytical Solution** is preferred when:
- Dataset is small to medium-sized
- You need the exact optimal solution
- Computational resources are not a constraint
- The feature matrix is well-conditioned

## Conclusion

Implementing linear regression from scratch provides deep insights into machine learning fundamentals. While production systems should use optimized libraries like scikit-learn, understanding the underlying mathematics and implementation details is invaluable for:

- Debugging machine learning pipelines
- Extending algorithms for specific use cases
- Understanding more complex algorithms that build on these foundations
- Making informed decisions about hyperparameters and model selection

The journey from mathematical formulation to working code illuminates the elegant simplicity underlying one of machine learning's most fundamental algorithms. Whether you choose gradient descent for its generalizability or the analytical solution for its precision, both approaches demonstrate how statistical learning can be translated into practical, working systems.

---

*Try implementing this code with your own datasets and experiment with different learning rates and regularization techniques. The insights gained from building algorithms from scratch will serve you well throughout your machine learning journey.*