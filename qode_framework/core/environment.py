"""
Environment classes for qODE simulations.

Environments define the propagation medium that affects wave behavior.
"""

import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap


class MediumType(Enum):
    """Medium types with their wave speed multipliers."""
    OPEN_SPACE = 1.0      # Parks, plazas - fastest propagation
    STREET = 0.7          # Roads, alleys - moderate speed
    BUILDING = 0.1        # Dense structures - very slow (mostly reflection)
    WATER = 0.85          # Rivers, lakes - slightly slower than open
    VEGETATION = 0.5      # Trees, gardens - moderate attenuation
    CUSTOM = 0.5          # User-defined medium


class BaseEnvironment(ABC):
    """
    Abstract base class for all environments.

    Subclass this to create custom environments (e.g., indoor, terrain, etc.)
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = np.ones((height, width))  # Speed multiplier grid
        self.medium_map = np.zeros((height, width), dtype=int)  # Medium type indices

    @abstractmethod
    def _generate_layout(self):
        """Generate the environment layout. Override in subclasses."""
        pass

    def get_speed_multiplier(self, x: float, y: float) -> float:
        """Get the wave speed multiplier at a given location."""
        ix = int(np.clip(x, 0, self.width - 1))
        iy = int(np.clip(y, 0, self.height - 1))
        return self.grid[iy, ix]

    def get_speed_gradient(self, x: float, y: float) -> Tuple[float, float]:
        """Get the gradient of speed multiplier for refraction effects."""
        eps = 0.5
        dx = (self.get_speed_multiplier(x + eps, y) -
              self.get_speed_multiplier(x - eps, y)) / (2 * eps)
        dy = (self.get_speed_multiplier(x, y + eps) -
              self.get_speed_multiplier(x, y - eps)) / (2 * eps)
        return dx, dy

    def get_medium_name(self, x: float, y: float) -> str:
        """Get the medium name at a given location."""
        ix = int(np.clip(x, 0, self.width - 1))
        iy = int(np.clip(y, 0, self.height - 1))
        medium_val = self.medium_map[iy, ix]
        medium_names = ['Open Space', 'Street', 'Building', 'Water', 'Vegetation']
        return medium_names[medium_val] if medium_val < len(medium_names) else 'Unknown'

    @abstractmethod
    def visualize(self, ax=None, show_legend: bool = True):
        """Visualize the environment."""
        pass


class UrbanEnvironment(BaseEnvironment):
    """
    Urban environment with buildings, streets, parks, water bodies, and vegetation.

    Parameters:
        width: Grid width
        height: Grid height
        seed: Random seed for reproducible layouts
        street_spacing: Distance between parallel streets
        building_density: Number of buildings per block (1-5)
        park_size: Size of central park as fraction of grid
        include_river: Whether to include a river
    """

    def __init__(
        self,
        width: int = 100,
        height: int = 100,
        seed: int = 42,
        street_spacing: int = 15,
        building_density: int = 3,
        park_size: float = 0.2,
        include_river: bool = True
    ):
        super().__init__(width, height)

        self.seed = seed
        self.street_spacing = street_spacing
        self.building_density = building_density
        self.park_size = park_size
        self.include_river = include_river

        np.random.seed(seed)
        self._generate_layout()

    def _generate_layout(self):
        """Generate a realistic urban layout."""

        # Initialize with open space
        self.grid.fill(MediumType.OPEN_SPACE.value)
        self.medium_map.fill(0)

        # Create street grid
        self._add_streets()

        # Add buildings in blocks
        self._add_buildings()

        # Add central park
        self._add_park()

        # Add river if enabled
        if self.include_river:
            self._add_river()

    def _add_streets(self):
        """Add street grid."""
        for i in range(0, self.width, self.street_spacing):
            # Vertical streets
            self.grid[:, max(0, i-1):min(self.width, i+2)] = MediumType.STREET.value
            self.medium_map[:, max(0, i-1):min(self.width, i+2)] = 1

        for j in range(0, self.height, self.street_spacing):
            # Horizontal streets
            self.grid[max(0, j-1):min(self.height, j+2), :] = MediumType.STREET.value
            self.medium_map[max(0, j-1):min(self.height, j+2), :] = 1

    def _add_buildings(self):
        """Add buildings in city blocks."""
        for i in range(0, self.width - self.street_spacing, self.street_spacing):
            for j in range(0, self.height - self.street_spacing, self.street_spacing):
                num_buildings = np.random.randint(1, self.building_density + 1)

                for _ in range(num_buildings):
                    bw = np.random.randint(3, 8)
                    bh = np.random.randint(3, 8)
                    bx = i + 3 + np.random.randint(0, max(1, self.street_spacing - bw - 4))
                    by = j + 3 + np.random.randint(0, max(1, self.street_spacing - bh - 4))

                    bx = min(bx, self.width - bw)
                    by = min(by, self.height - bh)

                    self.grid[by:by+bh, bx:bx+bw] = MediumType.BUILDING.value
                    self.medium_map[by:by+bh, bx:bx+bw] = 2

    def _add_park(self):
        """Add central park with vegetation border."""
        park_w = int(self.width * self.park_size)
        park_h = int(self.height * self.park_size)
        park_x = (self.width - park_w) // 2
        park_y = (self.height - park_h) // 2

        # Park (open space)
        self.grid[park_y:park_y+park_h, park_x:park_x+park_w] = MediumType.OPEN_SPACE.value
        self.medium_map[park_y:park_y+park_h, park_x:park_x+park_w] = 0

        # Vegetation border
        veg_margin = 2
        for dy in range(-veg_margin, park_h + veg_margin):
            for dx in range(-veg_margin, park_w + veg_margin):
                y, x = park_y + dy, park_x + dx
                if 0 <= y < self.height and 0 <= x < self.width:
                    if (dy < 0 or dy >= park_h or dx < 0 or dx >= park_w):
                        if self.medium_map[y, x] == 0:
                            self.grid[y, x] = MediumType.VEGETATION.value
                            self.medium_map[y, x] = 4

    def _add_river(self):
        """Add a winding river."""
        river_y = int(self.height * 0.75)

        for x in range(self.width):
            river_width = 3 + int(2 * np.sin(x * 0.1))
            for dy in range(-river_width, river_width + 1):
                y = river_y + dy
                if 0 <= y < self.height:
                    self.grid[y, x] = MediumType.WATER.value
                    self.medium_map[y, x] = 3

    def visualize(self, ax=None, show_legend: bool = True):
        """Visualize the urban environment."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        # Custom colormap for urban features
        colors = ['#90EE90', '#808080', '#8B4513', '#4169E1', '#228B22']
        cmap = LinearSegmentedColormap.from_list('urban', colors, N=5)

        ax.imshow(self.medium_map, cmap=cmap, origin='lower',
                  extent=[0, self.width, 0, self.height])

        if show_legend:
            legend_elements = [
                patches.Patch(facecolor='#90EE90', label='Open Space'),
                patches.Patch(facecolor='#808080', label='Street'),
                patches.Patch(facecolor='#8B4513', label='Building'),
                patches.Patch(facecolor='#4169E1', label='Water'),
                patches.Patch(facecolor='#228B22', label='Vegetation'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Urban Environment')

        return ax

    def get_colormap(self):
        """Return the colormap used for visualization."""
        colors = ['#90EE90', '#808080', '#8B4513', '#4169E1', '#228B22']
        return LinearSegmentedColormap.from_list('urban', colors, N=5)
