import numpy as np
import torch
from torchlevy import LevyStable, stable_dist # a user-defined package for Levy stable distribution

def token_data_generate():
    np.random.seed(42)  # for reproducibility
    
    n_samples = 1000
    dim = 20
    
    # 1) Generate Bernoulli-based dataset
    #    - first coord ~ Bernoulli(0.9)
    #    - second coord ~ Bernoulli(0.5)
    #    - remaining 8 coords ~ Bernoulli(0.1)
    x1 = np.random.binomial(1, 0.9, size=(n_samples, 2))
    x2 = np.random.binomial(1, 0.5, size=(n_samples, 2))
    x_others = np.random.binomial(1, 0.1, size=(n_samples, dim - 4))
    
    X_token = np.hstack([x1, x2, x_others])  # shape (1000, 20)
    
    # 2) Define a random weight vector w (20-D) -- no bias
    w = np.random.normal(0, 1, size=dim)
    
    # 3) true label, noisy labels
    y_token = X_token @ w
    y_token_gaussian_one = y_token + np.random.normal(0, 1, size=n_samples)
    y_token_gaussian_three = y_token + np.random.normal(0, 3, size=n_samples)
    y_token_student_one = y_token + np.nan_to_num(np.random.standard_t(df=1, size=n_samples), nan=0.0) # cauchy, undefined mean and variance
    y_token_student_two = y_token + np.nan_to_num(np.random.standard_t(df=2, size=n_samples), nan=0.0) # student-t infinite variance
    
    
    # 4) Generate the normal-features dataset
    #    1000 samples, each 10-D from a standard normal
    X_normal = np.random.randn(n_samples, dim)
    
    # 5) true label, noisy labels
    y_normal = X_normal @ w
    y_normal_gaussian_one = y_normal + np.random.normal(0, 1, size=n_samples)
    y_normal_gaussian_three = y_normal + np.random.normal(0, 3, size=n_samples)
    y_normal_student_one = y_normal + np.nan_to_num(np.random.standard_t(df=1, size=n_samples), nan=0.0) # cauchy, undefined mean and variance
    y_normal_student_two = y_normal + np.nan_to_num(np.random.standard_t(df=2, size=n_samples), nan=0.0) # student-t infinite variance
    
    # 6) Save everything into one .npz file
    #    We'll store them under keys "X_bern", "y_bern", "X_normal", "y_normal".
    np.savez(
        "../datasets/syntoken_data.npz",
        w_true=w,
        X_token=X_token,
        y_token=y_token,
        y_token_gaussian_one=y_token_gaussian_one,
        y_token_gaussian_three=y_token_gaussian_three,
        y_token_student_one=y_token_student_one,
        y_token_student_two=y_token_student_two,
        X_normal=X_normal, 
        y_normal=y_normal,
        y_normal_gaussian_one=y_normal_gaussian_one,
        y_normal_gaussian_three=y_normal_gaussian_three,
        y_normal_student_one=y_normal_student_one,
        y_normal_student_two=y_normal_student_two,
    )
    
    print("Data saved to data.npz")
    print("Shapes:")
    print(" w:", w.shape)
    print("  X_token:", X_token.shape)
    print("  y_token:", y_token.shape)
    print("  X_normal:", X_normal.shape)
    print("  y_normal:", y_normal.shape)


if __name__ == "__main__":
    token_data_generate()