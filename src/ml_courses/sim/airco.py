"""
Air conditioning temperature preference simulator.

This module provides the AircoSimulator class, which simulates daily temperature preferences (β values)
over time with seasonal variations, and includes plotting and statistics utilities.
"""

from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray


class AircoSimulator:
    """
    Air conditioning temperature preference simulator.

    Simulates β values (temperature preferences) over time with seasonal variations.
    Winter preferences tend to be higher (warmer) with less variability,
    while summer preferences are lower (cooler) with more variability.
    """

    def __init__(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        winter_mean: float = 23.0,
        summer_mean: float = 21.0,
        winter_std: float = 1.2,
        summer_std: float = 2.0,
        seed: int = 42,
    ):
        """
        Initialize the AircoSimulator and generate simulation data.

        Args:
            start_date: Start date for simulation (default: 2022-01-01)
            end_date: End date for simulation (default: 2023-12-31)
            winter_mean: Mean temperature preference in winter (°C)
            summer_mean: Mean temperature preference in summer (°C)
            winter_std: Standard deviation in winter (°C)
            summer_std: Standard deviation in summer (°C)
            seed: Random seed for reproducibility
        """
        # Set default date range if not provided
        self.start_date = start_date or datetime(2022, 1, 1)
        self.end_date = end_date or datetime(2023, 12, 31)

        # Seasonal parameters
        self.winter_mean = winter_mean
        self.summer_mean = summer_mean
        self.winter_std = winter_std
        self.summer_std = summer_std

        # Random number generator
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Data storage - will be initialized in simulate()
        self._data: pd.DataFrame | None = None
        self._dates: pd.DatetimeIndex | None = None
        self._day_of_year: NDArray[np.int_] | None = None
        self._seasonal_mean: NDArray[np.float64] | None = None
        self._seasonal_std: NDArray[np.float64] | None = None

        # Generate simulation data immediately
        self.simulate()

    def _generate_dates(self) -> None:
        """Generate date range and day of year array."""
        self._dates = pd.date_range(start=self.start_date, end=self.end_date, freq="D")
        # Create day of year for seasonal calculations (1-365/366)
        self._day_of_year = np.array([d.timetuple().tm_yday for d in self._dates])

    def _calculate_seasonal_parameters(self) -> None:
        """Calculate seasonal mean and standard deviation using sinusoidal functions."""
        if self._day_of_year is None:
            self._generate_dates()

        # Ensure _day_of_year is not None after _generate_dates()
        assert self._day_of_year is not None, "Day of year should be initialized"

        # Create seasonal mean using sinusoidal function
        # Peak in winter (day ~15 = mid January), minimum in summer (day ~195 = mid July)
        self._seasonal_mean = self.summer_mean + (self.winter_mean - self.summer_mean) * 0.5 * (
            1 + np.cos(2 * np.pi * (self._day_of_year - 15) / 365)
        )

        # Create seasonal standard deviation
        # Minimum std in winter, maximum in summer
        self._seasonal_std = self.winter_std + (self.summer_std - self.winter_std) * 0.5 * (
            1 - np.cos(2 * np.pi * (self._day_of_year - 15) / 365)
        )

    def simulate(self) -> pd.DataFrame:
        """
        Run the complete simulation.

        Returns
        -------
            DataFrame with columns: date, beta, seasonal_mean, seasonal_std, month
        """
        # Generate dates and seasonal parameters
        self._generate_dates()
        self._calculate_seasonal_parameters()

        # Ensure all required attributes are initialized
        assert self._dates is not None, "Dates should be initialized"
        assert self._seasonal_mean is not None, "Seasonal mean should be initialized"
        assert self._seasonal_std is not None, "Seasonal std should be initialized"

        # Generate β values from Gaussian distributions with seasonal parameters
        beta_values = self.rng.normal(self._seasonal_mean, self._seasonal_std)

        # Create DataFrame
        self._data = pd.DataFrame(
            {
                "date": self._dates,
                "beta": beta_values,
                "seasonal_mean": self._seasonal_mean,
                "seasonal_std": self._seasonal_std,
                "month": [d.month for d in self._dates],
            }
        )

        return self._data

    def get_data(self) -> pd.DataFrame:
        """
        Get the simulation data.

        Returns
        -------
            DataFrame with simulation results
        """
        if self._data is None:
            raise RuntimeError("Simulation has not been run yet. Call simulate() first.")
        return self._data

    def get_statistics(self) -> dict[str, float]:
        """
        Get summary statistics of the simulation.

        Returns
        -------
            Dictionary with mean, std, min, max of β values
        """
        if self._data is None:
            raise RuntimeError("Simulation has not been run yet. Call simulate() first.")
        beta_values = self._data["beta"]
        return {
            "mean": float(np.mean(beta_values)),
            "std": float(np.std(beta_values)),
            "min": float(np.min(beta_values)),
            "max": float(np.max(beta_values)),
            "n_days": len(beta_values),
        }

    @property
    def parameters(self) -> dict[str, Any]:
        """
        Get all simulation parameters.

        Returns
        -------
            Dictionary with all configuration parameters
        """
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "winter_mean": self.winter_mean,
            "summer_mean": self.summer_mean,
            "winter_std": self.winter_std,
            "summer_std": self.summer_std,
            "seed": self.seed,
        }

    def plot_results(self, figsize: tuple = (12, 8)) -> None:
        """
        Plot the simulation results.

        Args:
            figsize: Figure size as (width, height)
        """
        if self._data is None:
            raise RuntimeError("Simulation has not been run yet. Call simulate() first.")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Plot β values over time
        ax1.plot(self._data["date"], self._data["beta"], alpha=0.7, linewidth=0.8)
        ax1.plot(
            self._data["date"],
            self._data["seasonal_mean"],
            color="red",
            linewidth=2,
            label="Seasonal Mean",
        )
        ax1.set_ylabel("Temperature Preference β (°C)")
        ax1.set_title("Air Conditioning Temperature Preferences Over Time")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot seasonal standard deviation
        ax2.plot(self._data["date"], self._data["seasonal_std"], color="orange", linewidth=2)
        ax2.set_ylabel("Standard Deviation (°C)")
        ax2.set_xlabel("Date")
        ax2.set_title("Seasonal Variability in Temperature Preferences")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def __repr__(self) -> str:
        """String representation of the simulator."""
        return (
            f"AircoSimulator(start_date={self.start_date.date()}, "
            f"end_date={self.end_date.date()}, "
            f"winter_mean={self.winter_mean}, "
            f"summer_mean={self.summer_mean})"
        )
