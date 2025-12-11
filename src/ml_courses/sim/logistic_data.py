"""Simulate data for logistic regression / binary classification."""

import numpy as np
import pandas as pd


def generate_binary_tipping_data(
    n_customers: int = 50,
    true_b1: float = -2.0,
    true_b2: float = 0.25,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Generate synthetic coffee shop tipping data for binary classification.

    Simulates a realistic scenario where customers either leave a tip (1) or
    don't (0) based on their order total. The relationship follows a logistic
    function: P(tip) = σ(b1 + b2 × order_total)

    Parameters
    ----------
    n_customers : int, default=50
        Number of customer observations to generate
    true_b1 : float, default=-2.0
        True intercept parameter (bias) for the logistic model
    true_b2 : float, default=0.25
        True slope parameter (effect of order size) for the logistic model
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    X : np.ndarray of shape (n_customers, 1)
        Feature matrix containing order totals
    y : np.ndarray of shape (n_customers,)
        Binary target vector (1 = tipped, 0 = no tip)
    metadata : dict
        Dictionary containing:
        - 'order_totals': original order totals
        - 'true_probs': true probabilities P(tip|order)
        - 'true_b1': true intercept parameter
        - 'true_b2': true slope parameter
        - 'n_tippers': count of customers who tipped
        - 'tip_rate': proportion of customers who tipped

    Examples
    --------
    >>> X, y, meta = generate_binary_tipping_data(n_customers=100, seed=42)
    >>> len(y)
    100
    >>> "tip_rate" in meta
    True
    >>> 0 <= meta["tip_rate"] <= 1
    True
    """
    rng = np.random.default_rng(seed)

    # Generate realistic order totals ($3 to $25)
    order_totals = rng.uniform(3, 25, n_customers)
    order_totals = np.sort(order_totals)  # Sort for nicer visualization

    # Calculate true probabilities using logistic function
    logits = true_b1 + true_b2 * order_totals
    true_probs = 1 / (1 + np.exp(-logits))

    # Generate binary outcomes based on these probabilities
    tips_binary = rng.binomial(n=1, p=true_probs, size=n_customers)

    # Prepare feature matrix (just order totals for now)
    X = order_totals.reshape(-1, 1)
    y = tips_binary

    # Metadata for analysis
    metadata = {
        "order_totals": order_totals,
        "true_probs": true_probs,
        "true_b1": true_b1,
        "true_b2": true_b2,
        "n_tippers": int(y.sum()),
        "tip_rate": float(y.mean()),
    }

    return X, y, metadata


def generate_binary_tipping_dataframe(
    n_customers: int = 50,
    true_b1: float = -2.0,
    true_b2: float = 0.25,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Generate synthetic tipping data and return as a pandas DataFrame.

    Convenience function that wraps generate_binary_tipping_data() and
    returns the results as a DataFrame instead of numpy arrays.

    Parameters
    ----------
    n_customers : int, default=50
        Number of customer observations to generate
    true_b1 : float, default=-2.0
        True intercept parameter
    true_b2 : float, default=0.25
        True slope parameter
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns:
        - 'order_total': order amount in dollars
        - 'tipped': binary indicator (1=tipped, 0=no tip)
        - 'true_prob': true probability of tipping
    metadata : dict
        Dictionary with true parameters and summary statistics

    Examples
    --------
    >>> df, meta = generate_binary_tipping_dataframe(n_customers=100, seed=42)
    >>> len(df)
    100
    >>> list(df.columns)
    ['order_total', 'tipped', 'true_prob']
    >>> bool(df["tipped"].isin([0, 1]).all())
    True
    """
    X, y, metadata = generate_binary_tipping_data(n_customers, true_b1, true_b2, seed)

    df = pd.DataFrame(
        {
            "order_total": metadata["order_totals"],
            "tipped": y,
            "true_prob": metadata["true_probs"],
        }
    )

    return df, metadata
