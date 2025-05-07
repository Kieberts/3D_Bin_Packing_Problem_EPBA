import math
import random
from typing import List, Tuple, Set, Optional, Dict

# Define constants for clarity (adjust values based on specific problem parameters)
DIMENSIONS = 3
X, Y, Z = 0, 1, 2

class Item:
    """
    Represents an item to be packed.

    Attributes:
        id (int): Unique identifier for the item.
        size (Tuple[float, float, float]): Original dimensions (length, width, height).
        weight (float): Weight of the item.
        is_rotatable (bool): Whether the item can be rotated.
        is_tiltable (bool): Whether the item can be tilted.
        is_stackable (bool): Whether other items can be placed on top.
        current_size (Tuple[float, float, float]): Dimensions after applying orientation.
        orientation (Tuple[str, int]): Current orientation ('not_tilted', 'tilted_x', 'tilted_y', rotation 0/1).
        position (Optional[Tuple[float, float, float]]): Position (corner closest to origin) if loaded. None otherwise.
    """
    def __init__(self, id: int, size: Tuple[float, float, float], weight: float,
                 is_rotatable: bool, is_tiltable: bool, is_stackable: bool):
        self.id = id
        self.original_size = size # Store original size
        self.weight = weight
        self.is_rotatable = is_rotatable
        self.is_tiltable = is_tiltable
        self.is_stackable = is_stackable
        self.current_size = size # Initial size is the original size
        self.orientation = ('not_tilted', 0) # Default orientation
        self.position: Optional[Tuple[float, float, float]] = None # (x, y, z) of the corner closest to origin

    def set_orientation(self, orientation: Tuple[str, int]):
        """
        Sets the item's orientation and updates its current_size.
        This is a simplified representation. A real implementation would need
        precise geometry calculations for tilting.
        """
        self.orientation = orientation
        tilt, rotation = orientation
        l, w, h = self.original_size

        # --- Simplified Orientation Logic ---
        # A full implementation requires careful handling of geometry,
        # especially for tilting. This example primarily handles rotation.
        if tilt == 'tilted_x':
             # Example: Tilt around x-axis (swap w and h) - highly simplified
             l, w, h = l, h, w
        elif tilt == 'tilted_y':
             # Example: Tilt around y-axis (swap l and h) - highly simplified
             l, w, h = h, w, l

        if rotation == 1 and self.is_rotatable:
            # Rotate around z-axis (swap l and w)
            self.current_size = (w, l, h)
        else:
            self.current_size = (l, w, h)
        # --- End Simplified Orientation Logic ---

    def get_possible_orientations(self) -> List[Tuple[str, int]]:
        """Returns a list of possible orientations based on item properties."""
        orientations = []
        tilts = ['not_tilted']
        if self.is_tiltable:
            # Simplification: Assume tiltable means rotatable too (as per PDF)
            # Add specific tilt options if needed
            tilts.extend(['tilted_x', 'tilted_y']) # Placeholder names

        rotations = [0]
        if self.is_rotatable or self.is_tiltable:
            rotations.append(1)

        for t in tilts:
            for r in rotations:
                # Avoid duplicates if rotation doesn't change dimensions
                temp_item = Item(self.id, self.original_size, self.weight, self.is_rotatable, self.is_tiltable, self.is_stackable)
                temp_item.set_orientation((t, r))
                current_orientation = (t,r)
                # Check if this orientation is effectively unique
                is_unique = True
                for existing_orientation_tuple in orientations:
                     existing_orientation, existing_size = existing_orientation_tuple
                     if existing_size == temp_item.current_size:
                         is_unique = False
                         break
                if is_unique:
                     orientations.append((current_orientation, temp_item.current_size))


        # Return only the orientation tuples
        return [o[0] for o in orientations]


    def get_end_position(self) -> Optional[Tuple[float, float, float]]:
        """Calculates the corner furthest from the origin if the item is placed."""
        if self.position is None:
            return None
        return (self.position[X] + self.current_size[X],
                self.position[Y] + self.current_size[Y],
                self.position[Z] + self.current_size[Z])

    def __repr__(self):
        return (f"Item(id={self.id}, size={self.original_size}, weight={self.weight}, "
                f"stackable={self.is_stackable}, pos={self.position}, orient={self.orientation})")

class ULD:
    """
    Represents a Unit Load Device (Container/Bin).

    Attributes:
        id (int): Unique identifier for the ULD.
        vertices (List[Tuple[float, float, float]]): List of vertex coordinates defining the shape.
        facets (List[List[int]]): List of facets, each defined by indices of its vertices.
                                    (Assumed convex, simple shapes for this example).
        weight_capacity (float): Maximum allowed weight.
        volume_capacity (float): Total volume.
        edge_width (float): Width of the edge where loading is restricted at the bottom.
        vertical_edge_offset (float): Minimum height to overlap the edge.
        use_substructure_allowed (bool): Whether a substructure can be used.
        bounding_box (Tuple[float, float, float]): Dimensions (LxWxH) of the ULD's bounding box.
        loaded_items (List[Item]): List of items currently loaded in the ULD.
        dummy_items (List[Item]): List of dummy items representing blocked space (edge, substructure).
        geometric_center (Tuple[float, float]): Geometric center in x-y plane.
    """
    def __init__(self, id: int, vertices: List[Tuple[float, float, float]],
                 facets: List[List[int]], weight_capacity: float, volume_capacity: float,
                 edge_width: float = 0.0, vertical_edge_offset: float = 0.0,
                 use_substructure_allowed: bool = False):
        self.id = id
        self.vertices = vertices
        self.facets = facets # Simplified representation - assumes simple shapes
        self.weight_capacity = weight_capacity
        self.volume_capacity = volume_capacity
        self.edge_width = edge_width
        self.vertical_edge_offset = vertical_edge_offset
        self.use_substructure_allowed = use_substructure_allowed
        self.bounding_box = self._calculate_bounding_box()
        self.geometric_center = (self.bounding_box[X] / 2, self.bounding_box[Y] / 2)
        self.loaded_items: List[Item] = []
        self.dummy_items: List[Item] = [] # For edge/substructure blocking

    def _calculate_bounding_box(self) -> Tuple[float, float, float]:
        """Calculates the bounding box dimensions based on vertices."""
        if not self.vertices:
            return (0.0, 0.0, 0.0)
        min_x = min(v[X] for v in self.vertices)
        max_x = max(v[X] for v in self.vertices)
        min_y = min(v[Y] for v in self.vertices)
        max_y = max(v[Y] for v in self.vertices)
        min_z = min(v[Z] for v in self.vertices)
        max_z = max(v[Z] for v in self.vertices)
        # Assuming origin is at (min_x, min_y, min_z) after shifting
        return (max_x - min_x, max_y - min_y, max_z - min_z)

    def get_total_weight(self) -> float:
        """Calculates the total weight of loaded items."""
        return sum(item.weight for item in self.loaded_items)

    def get_total_volume(self) -> float:
        """Calculates the total volume of loaded items."""
        return sum(item.current_size[X] * item.current_size[Y] * item.current_size[Z]
                   for item in self.loaded_items)

    def calculate_center_of_gravity(self) -> Optional[Tuple[float, float]]:
        """Calculates the Center of Gravity (CoG) in the x-y plane."""
        total_weight = 0.0
        weighted_sum_x = 0.0
        weighted_sum_y = 0.0

        if not self.loaded_items:
            return None

        for item in self.loaded_items:
            if item.position is None: continue # Should not happen if item is loaded
            item_center_x = item.position[X] + item.current_size[X] / 2
            item_center_y = item.position[Y] + item.current_size[Y] / 2
            total_weight += item.weight
            weighted_sum_x += item.weight * item_center_x
            weighted_sum_y += item.weight * item_center_y

        if total_weight == 0:
            return self.geometric_center # Or None, depending on desired behavior

        cog_x = weighted_sum_x / total_weight
        cog_y = weighted_sum_y / total_weight
        return (cog_x, cog_y)

    def __repr__(self):
        return (f"ULD(id={self.id}, bbox={self.bounding_box}, "
                f"items={len(self.loaded_items)}, weight={self.get_total_weight():.2f}/{self.weight_capacity}, "
                f"vol={self.get_total_volume():.2f}/{self.volume_capacity})")

class GridBasedAccelerator:
    """
    Accelerates collision and support checks using a spatial grid.
    (Based on Section 4.4.4)
    """
    def __init__(self, uld_bbox: Tuple[float, float, float], items: List[Item]):
        self.uld_bbox = uld_bbox
        self.cell_size = self._calculate_average_cell_size(items)
        if self.cell_size <= 0:
             print("Warning: Cannot create grid with non-positive cell size. Using default.")
             self.cell_size = min(d for d in uld_bbox if d > 0) if any(d > 0 for d in uld_bbox) else 1.0 # Fallback

        # Ensure cell_size is not zero to avoid division by zero
        if self.cell_size == 0:
            self.cell_size = 1.0 # Or some other small default value

        self.grid_dimensions = (
            math.ceil(uld_bbox[X] / self.cell_size) if self.cell_size > 0 else 1,
            math.ceil(uld_bbox[Y] / self.cell_size) if self.cell_size > 0 else 1,
            math.ceil(uld_bbox[Z] / self.cell_size) if self.cell_size > 0 else 1,
        )
        # Use a dictionary for sparse storage: key=(ix, iy, iz), value=Set[Item]
        self.grid: Dict[Tuple[int, int, int], Set[Item]] = {}
        self.epsilon = 1e-6 # Small value for floating point comparisons

    def _calculate_average_cell_size(self, items: List[Item]) -> float:
        """Calculates the average item edge size."""
        if not items:
            return min(d for d in self.uld_bbox if d > 0) if any(d > 0 for d in self.uld_bbox) else 1.0 # Fallback if no items
        total_edge_sum = sum(sum(item.original_size) for item in items)
        avg_size = total_edge_sum / (3 * len(items))
        # Ensure cell size is not larger than the smallest ULD dimension
        min_uld_dim = min(d for d in self.uld_bbox if d > 0) if any(d > 0 for d in self.uld_bbox) else 1.0
        return max(1.0, min(avg_size, min_uld_dim)) # Ensure cell size is at least 1.0

    def _get_grid_indices(self, coord: float, dimension_index: int) -> int:
        """Calculates the grid index for a coordinate along a dimension."""
        if self.cell_size <= 0: return 0
        idx = math.floor(coord / self.cell_size)
        # Clamp index to be within grid bounds
        return max(0, min(idx, self.grid_dimensions[dimension_index] - 1))

    def _get_intersecting_cells_indices(self, item_pos: Tuple[float, float, float],
                                       item_size: Tuple[float, float, float]) -> Set[Tuple[int, int, int]]:
        """Determines the indices of grid cells intersected by an item."""
        indices = set()
        if self.cell_size <= 0: return indices

        min_ix = self._get_grid_indices(item_pos[X], X)
        max_ix = self._get_grid_indices(item_pos[X] + item_size[X] - self.epsilon, X)
        min_iy = self._get_grid_indices(item_pos[Y], Y)
        max_iy = self._get_grid_indices(item_pos[Y] + item_size[Y] - self.epsilon, Y)
        min_iz = self._get_grid_indices(item_pos[Z], Z)
        max_iz = self._get_grid_indices(item_pos[Z] + item_size[Z] - self.epsilon, Z)

        for ix in range(min_ix, max_ix + 1):
            for iy in range(min_iy, max_iy + 1):
                for iz in range(min_iz, max_iz + 1):
                    indices.add((ix, iy, iz))
        return indices

    def add_item(self, item: Item):
        """Registers a loaded item in the grid cells it intersects."""
        if item.position is None: return
        indices = self._get_intersecting_cells_indices(item.position, item.current_size)
        for index_tuple in indices:
            if index_tuple not in self.grid:
                self.grid[index_tuple] = set()
            self.grid[index_tuple].add(item)

    def remove_item(self, item: Item):
         """Removes an item from the grid (e.g., during backtracking or refinement)."""
         if item.position is None: return
         indices = self._get_intersecting_cells_indices(item.position, item.current_size)
         for index_tuple in indices:
             if index_tuple in self.grid:
                 self.grid[index_tuple].discard(item)
                 if not self.grid[index_tuple]: # Remove empty set
                      del self.grid[index_tuple]


    def get_potential_colliders(self, item_to_place: Item, position: Tuple[float, float, float]) -> Set[Item]:
        """Gets items in potentially colliding grid cells."""
        potential_colliders = set()
        indices = self._get_intersecting_cells_indices(position, item_to_place.current_size)
        for index_tuple in indices:
            if index_tuple in self.grid:
                potential_colliders.update(self.grid[index_tuple])
        potential_colliders.discard(item_to_place) # Exclude self
        return potential_colliders

    def get_potential_supporters(self, item_to_place: Item, position: Tuple[float, float, float],
                                 max_padding_height: float) -> Set[Item]:
        """Gets items in grid cells potentially supporting the new item."""
        potential_supporters = set()
        # Define the search volume below the item
        support_pos = (position[X], position[Y], max(0.0, position[Z] - max_padding_height - self.epsilon))
        support_size = (item_to_place.current_size[X], item_to_place.current_size[Y], max_padding_height + self.epsilon)

        indices = self._get_intersecting_cells_indices(support_pos, support_size)

        for index_tuple in indices:
            if index_tuple in self.grid:
                 # Further filter: only consider items whose top surface is within range
                 for item in self.grid[index_tuple]:
                      if item.position is None: continue
                      item_top_z = item.position[Z] + item.current_size[Z]
                      # Item's top must be between pos_z - padding and pos_z
                      if position[Z] - max_padding_height - self.epsilon <= item_top_z <= position[Z] + self.epsilon:
                           potential_supporters.add(item)

        potential_supporters.discard(item_to_place) # Exclude self
        return potential_supporters

# --- Core Algorithm Functions ---

def adapt_uld(uld: ULD, use_substructure: bool) -> None:
    """
    Adds dummy items to represent ULD edge restrictions and optional substructure.
    (Based on Section 4.1)

    Args:
        uld: The ULD object to adapt.
        use_substructure: Boolean indicating if a substructure should be added.
    """
    uld.dummy_items = [] # Clear previous dummy items
    ew = uld.edge_width
    delta = uld.vertical_edge_offset
    L, W, H = uld.bounding_box

    if ew > 0 and delta > 0:
        # Dummy item height slightly less than delta to allow overlap check at delta
        dummy_h = max(0, delta - 1e-6)

        # Create 4 non-stackable dummy items for the edges
        # Front edge
        uld.dummy_items.append(Item(id=-1, size=(L, ew, dummy_h), weight=0, is_rotatable=False, is_tiltable=False, is_stackable=False))
        uld.dummy_items[-1].position = (0, 0, 0)
        # Back edge
        uld.dummy_items.append(Item(id=-2, size=(L, ew, dummy_h), weight=0, is_rotatable=False, is_tiltable=False, is_stackable=False))
        uld.dummy_items[-1].position = (0, W - ew, 0)
        # Left edge (excluding corners covered by front/back)
        uld.dummy_items.append(Item(id=-3, size=(ew, W - 2 * ew, dummy_h), weight=0, is_rotatable=False, is_tiltable=False, is_stackable=False))
        uld.dummy_items[-1].position = (0, ew, 0)
        # Right edge (excluding corners covered by front/back)
        uld.dummy_items.append(Item(id=-4, size=(ew, W - 2 * ew, dummy_h), weight=0, is_rotatable=False, is_tiltable=False, is_stackable=False))
        uld.dummy_items[-1].position = (L - ew, ew, 0)

    if use_substructure and uld.use_substructure_allowed and delta > 0:
        # Add one stackable dummy item for the substructure (height = delta)
        # Covering the floor area inside the edges
        sub_l = L - 2 * ew
        sub_w = W - 2 * ew
        if sub_l > 0 and sub_w > 0:
            uld.dummy_items.append(Item(id=-5, size=(sub_l, sub_w, delta), weight=0, is_rotatable=False, is_tiltable=False, is_stackable=True))
            uld.dummy_items[-1].position = (ew, ew, 0)

def group_and_sort_items(items: List[Item], sort_criterion: str, randomization_degree: float) -> List[Tuple[Item, List[Tuple[str, int]]]]:
    """
    Groups items, sorts them based on criteria, and determines orientations.
    Returns an ordered list: [(item1, [orientations]), (item2, [orientations]), ...].
    (Based on Section 4.2)

    Args:
        items: List of items to be loaded.
        sort_criterion: 'cumulated_volume', 'highest_volume',
                         'stackability-cumulated_volume', 'stackability-highest_volume', 'random'.
        randomization_degree (rho): Value between 0 and 1 for randomization level.

    Returns:
        List of tuples, where each tuple contains an item and a list of its
        relevant orientations for the current similar group context.
    """
    # 1. Group identical items
    identical_groups: Dict[Tuple, List[Item]] = {}
    for item in items:
        # Key: (tuple(sorted dimensions), weight, rotatable, tiltable, stackable)
        key = (tuple(sorted(item.original_size)), item.weight, item.is_rotatable, item.is_tiltable, item.is_stackable)
        if key not in identical_groups:
            identical_groups[key] = []
        identical_groups[key].append(item)

    # 2. Group similar items (based on possible heights and stackability)
    similar_groups: Dict[Tuple[float, bool], List[List[Item]]] = {} # Key: (height, is_stackable), Value: List of identical_groups
    processed_identical_groups = set()

    for key, identical_group in identical_groups.items():
        item_example = identical_group[0] # Use first item as representative
        possible_heights = set()
        # Simplified: Check original dimensions and potentially swapped dimensions if tiltable
        possible_heights.add(item_example.original_size[Z])
        if item_example.is_tiltable:
             possible_heights.add(item_example.original_size[X]) # Height if tilted around Y
             possible_heights.add(item_example.original_size[Y]) # Height if tilted around X

        stackable = item_example.is_stackable
        group_tuple = tuple(sorted(key)) # Use a hashable representation of the identical group key

        if group_tuple in processed_identical_groups:
             continue

        for h in possible_heights:
            similar_key = (h, stackable)
            if similar_key not in similar_groups:
                similar_groups[similar_key] = []
            # Add the whole identical group list
            similar_groups[similar_key].append(identical_group)
        processed_identical_groups.add(group_tuple)


    # 3. Sort similar groups and identical groups within them
    def get_sort_key(group_list: List[Item], criterion: str):
        """Helper to get sorting key for an identical item group."""
        if not group_list: return 0
        item_example = group_list[0]
        vol = item_example.original_size[X] * item_example.original_size[Y] * item_example.original_size[Z]
        cum_vol = vol * len(group_list)

        if criterion == 'cumulated_volume':
            return cum_vol
        elif criterion == 'highest_volume':
            return vol # All items in group have same volume
        elif criterion == 'stackability-cumulated_volume':
            # Sort by stackable (True=1, False=0) first (desc), then cum_vol (desc)
            return (item_example.is_stackable, cum_vol)
        elif criterion == 'stackability-highest_volume':
             # Sort by stackable (True=1, False=0) first (desc), then vol (desc)
             return (item_example.is_stackable, vol)
        elif criterion == 'random':
            return random.random()
        else:
            return cum_vol # Default

    # Sort the list of similar group keys
    sorted_similar_keys = sorted(
        similar_groups.keys(),
        key=lambda sk: get_sort_key(similar_groups[sk][0], sort_criterion), # Use first identical group in similar group for sorting similar groups
        reverse=True
    )

    # Sort identical groups within each similar group
    for sk in sorted_similar_keys:
        similar_groups[sk].sort(
            key=lambda ig: get_sort_key(ig, sort_criterion),
            reverse=True
        )

    # 4. Apply randomization (adapted from Shaw (1997))
    def randomize_list(original_list: list, rho: float, maintain_stackability_order: bool = False) -> list:
        """Applies randomization to a list."""
        if rho <= 0 or rho > 1: return original_list
        if not original_list: return []

        if maintain_stackability_order:
             # Split into stackable and non-stackable, randomize separately, then combine
             stackable_groups = [g for g in original_list if g[0][0].is_stackable] # Check first item of first identical group
             non_stackable_groups = [g for g in original_list if not g[0][0].is_stackable]
             randomized_stackable = randomize_list(stackable_groups, rho, False)
             randomized_non_stackable = randomize_list(non_stackable_groups, rho, False)
             return randomized_stackable + randomized_non_stackable # Stackable first

        n = len(original_list)
        remaining_list = original_list[:]
        randomized = []
        for j in range(n):
            y_j = random.random()
            # Calculate index based on Shaw's formula (adapted)
            # Ensure exponent is non-negative
            power_val = 1.0 / rho if rho > 0 else 1.0 # Avoid division by zero or negative exponent issues
            index = math.floor(y_j ** power_val * (len(remaining_list)))
            index = min(index, len(remaining_list) - 1) # Clamp index
            randomized.append(remaining_list.pop(index))
        return randomized

    # Randomize similar groups list
    is_stackability_sort = sort_criterion.startswith('stackability')
    randomized_keys = randomize_list(sorted_similar_keys, randomization_degree, is_stackability_sort)

    # Randomize identical groups within each similar group
    randomized_similar_groups = {}
    for sk in randomized_keys:
        randomized_similar_groups[sk] = randomize_list(similar_groups[sk], randomization_degree, False) # No stackability constraint within similar group

    # 5. Create the final ordered list with relevant orientations
    final_ordered_list = []
    processed_items = set() # Keep track of items already added to the final list for a specific pass

    for height, stackable in randomized_keys:
        similar_key = (height, stackable)
        for identical_group in randomized_similar_groups[similar_key]:
            for item in identical_group:
                 # Determine orientations that result in the target height 'h'
                 relevant_orientations = []
                 possible_orientations = item.get_possible_orientations() # Get all valid orientations

                 for orient in possible_orientations:
                      # Temporarily apply orientation to check resulting height
                      original_current_size = item.current_size
                      original_orientation = item.orientation
                      item.set_orientation(orient)
                      # Use a tolerance for floating point comparison
                      if abs(item.current_size[Z] - height) < 1e-6:
                           relevant_orientations.append(orient)
                      # Restore original state
                      item.current_size = original_current_size
                      item.orientation = original_orientation

                 if relevant_orientations and item.id not in processed_items : # Check if item needs processing in this pass
                      final_ordered_list.append((item, relevant_orientations))
                      processed_items.add(item.id) # Mark as processed for this height group pass

        processed_items.clear() # Reset for the next similar group height


    return final_ordered_list


def get_next_extreme_point(extreme_points: Set[Tuple[float, float, float]]) -> Optional[Tuple[float, float, float]]:
    """
    Selects the next extreme point to try, sorted z, y, x.
    (Based on Section 4.3, Figure 4c)

    Args:
        extreme_points: The set of current candidate points.

    Returns:
        The next extreme point tuple (x, y, z) or None if the set is empty.
    """
    if not extreme_points:
        return None
    # Sort ascending by z, then y, then x
    # Convert set to list for sorting
    sorted_points = sorted(list(extreme_points), key=lambda p: (p[Z], p[Y], p[X]))
    # In a real implementation, you'd likely iterate through this sorted list
    # and remove points as they are used or deemed invalid.
    # This function just returns the 'best' according to the sort order.
    return sorted_points[0]

def move_extreme_point(ep: Tuple[float, float, float], item_size: Tuple[float, float, float], uld: ULD) -> Tuple[float, float, float]:
    """
    Checks if an extreme point needs moving due to tilted ULD facets and calculates the new position.
    (Based on Section 4.3) - Simplified version assuming specific facet orientations.

    Args:
        ep: The original extreme point (x, y, z).
        item_size: The size (lx, ly, lz) of the item to be placed.
        uld: The ULD object containing facet information.

    Returns:
        The potentially moved extreme point (x, y, z). Returns original ep if no move is needed/possible.
    """
    moved_ep = ep
    min_move_dist = 0.0

    # --- Simplified Facet Check ---
    # This checks only for the specific tilted facets described in Fig 5a/5c
    # A full implementation needs to check all facets or have a more robust way
    # to identify "critical" facets near the extreme point.

    # Example check for a facet like Fig 5a/5c (Simplified: plane n2*y + n3*z = a, with n2>0, n3<=0)
    # This requires knowing the plane equations of the ULD facets.
    # Let's assume we have a function get_critical_facet(ep, uld) -> Optional[plane_equation]
    # plane_equation = (n1, n2, n3, a) where n1*x + n2*y + n3*z = a

    # Placeholder: Assume a critical facet exists for demonstration
    # In reality, you'd iterate through uld.facets, calculate plane equations,
    # and check if 'ep' lies on a facet matching the criteria n1=0, n2>0, n3<=0.
    critical_facet_found = False
    n1, n2, n3, a = 0.0, 0.0, 0.0, 0.0 # Placeholder for plane equation parameters

    # --- Find the relevant critical facet ---
    # This logic needs the actual facet definitions and plane equations.
    # Example: Iterate through facets, check if ep lies on it, check normal vector.
    # For simplicity, let's assume we identified one critical facet:
    # critical_facet_found = True
    # n1, n2, n3, a = 0.0, 1.0, -0.5, uld.bounding_box[Y] # Example plane y - 0.5z = W

    if critical_facet_found and n2 > 0: # Ensure n2 is positive to avoid division by zero/negative move
        # Calculate required move distance 'nu' (ν in the PDF)
        # We need the top-left corner of the item relative to ep: (ep_x + sx, ep_y + nu, ep_z + sz)
        # Equation: n1*(ep_x + sx) + n2*(ep_y + nu) + n3*(ep_z + sz) = a
        # Since n1=0: n2*(ep_y + nu) + n3*(ep_z + sz) = a
        # n2*nu = a - n2*ep_y - n3*(ep_z + item_size[Z])
        nu = (a - n2 * ep[Y] - n3 * (ep[Z] + item_size[Z])) / n2

        if nu > 1e-6: # If positive move is required (use tolerance)
            moved_ep = (ep[X], ep[Y] + nu, ep[Z])
            # print(f"Debug: Moving EP {ep} by {nu} in Y for item size {item_size} due to critical facet.") # Debug print

    return moved_ep

def check_uld_fit(item: Item, pos: Tuple[float, float, float], uld: ULD) -> bool:
    """
    Checks if the item, placed at pos, fits entirely within the ULD boundaries.
    Handles bounding box and potentially tilted facets.
    (Based on Section 4.4.1)
    """
    L, W, H = uld.bounding_box
    item_end_pos = (pos[X] + item.current_size[X], pos[Y] + item.current_size[Y], pos[Z] + item.current_size[Z])
    epsilon = 1e-6 # Tolerance for floating point comparisons

    # 1. Bounding Box Check (Essential)
    if (pos[X] < -epsilon or pos[Y] < -epsilon or pos[Z] < -epsilon or
            item_end_pos[X] > L + epsilon or
            item_end_pos[Y] > W + epsilon or
            item_end_pos[Z] > H + epsilon):
        # print(f"Debug: Failed BBox check: pos={pos}, end={item_end_pos}, bbox={uld.bounding_box}")
        return False

    # 2. Tilted Facet Check (Simplified)
    # A full implementation requires iterating through ULD facets, calculating plane equations,
    # and checking if the item's furthest vertex in the direction of the facet normal lies inside.
    # Plane equation: n1*x + n2*y + n3*z = a (normal points inwards)
    # Check condition: n . (corner_point) >= a
    # The corner point to check depends on the signs of n1, n2, n3.
    # Example: if n1<0, n2<0, n3<0, check item_end_pos. If n1>0, n2>0, n3>0, check pos.

    # Placeholder: Assume ULD is cuboid for simplicity here.
    # Add checks for specific tilted facets if needed, similar to move_extreme_point logic.

    # 3. ULD Edge Check (Section 4.1 / Constraint)
    if uld.edge_width > 0 and pos[Z] < uld.vertical_edge_offset - epsilon:
        ew = uld.edge_width
        # Check if item overlaps the restricted edge area at low height
        overlaps_left = pos[X] < ew - epsilon
        overlaps_right = item_end_pos[X] > L - ew + epsilon
        overlaps_front = pos[Y] < ew - epsilon
        overlaps_back = item_end_pos[Y] > W - ew + epsilon

        # Check x-edges
        if (overlaps_left or overlaps_right) and (pos[Y] < W - ew + epsilon and item_end_pos[Y] > ew - epsilon):
             # print(f"Debug: Failed Edge check (X): pos={pos}, end={item_end_pos}, Z={pos[Z]} < {uld.vertical_edge_offset}")
             return False
         # Check y-edges
        if (overlaps_front or overlaps_back) and (pos[X] < L - ew + epsilon and item_end_pos[X] > ew - epsilon):
             # print(f"Debug: Failed Edge check (Y): pos={pos}, end={item_end_pos}, Z={pos[Z]} < {uld.vertical_edge_offset}")
             return False


    return True

def check_collision(item_to_place: Item, pos: Tuple[float, float, float],
                    potential_colliders: Set[Item]) -> bool:
    """
    Checks if the item at the given position collides with any existing items.
    (Based on Section 4.4.2)

    Args:
        item_to_place: The new item.
        pos: The potential position for the new item.
        potential_colliders: A set of already loaded items to check against (can be pre-filtered by the grid).

    Returns:
        True if no collision occurs, False otherwise.
    """
    s_new = item_to_place.current_size
    e_new = pos
    epsilon = 1e-6 # Tolerance

    for loaded_item in potential_colliders:
        if loaded_item.position is None: continue # Should not happen

        s_loaded = loaded_item.current_size
        e_loaded = loaded_item.position

        # Check for overlap in all three dimensions
        overlap_x = (e_new[X] < e_loaded[X] + s_loaded[X] - epsilon) and \
                    (e_loaded[X] < e_new[X] + s_new[X] - epsilon)
        overlap_y = (e_new[Y] < e_loaded[Y] + s_loaded[Y] - epsilon) and \
                    (e_loaded[Y] < e_new[Y] + s_new[Y] - epsilon)
        overlap_z = (e_new[Z] < e_loaded[Z] + s_loaded[Z] - epsilon) and \
                    (e_loaded[Z] < e_new[Z] + s_new[Z] - epsilon)

        if overlap_x and overlap_y and overlap_z:
            # print(f"Debug: Collision detected between Item {item_to_place.id} at {pos} and Item {loaded_item.id} at {e_loaded}")
            return False # Collision detected
    return True # No collision

def check_non_floating_and_stackability(item_to_place: Item, pos: Tuple[float, float, float],
                                        potential_supporters: Set[Item], uld_bbox: Tuple[float, float, float],
                                        min_overlap_ratio: float, max_padding_height: float) -> bool:
    """
    Checks if the item is sufficiently supported and doesn't violate stackability.
    Implements Algorithm 2 from the PDF (with simplifications).

    Args:
        item_to_place: The new item.
        pos: The potential position (x, y, z).
        potential_supporters: Set of items (and potentially dummy floor) below the new item.
        uld_bbox: Dimensions of the ULD.
        min_overlap_ratio (o): Minimum required support area ratio.
        max_padding_height (ħ): Maximum vertical gap fillable by padding.

    Returns:
        True if the placement is valid (supported and respects stackability), False otherwise.
    """
    epsilon = 1e-6
    item_base_area = item_to_place.current_size[X] * item_to_place.current_size[Y]
    if item_base_area <= 0: return True # Item has no base area, cannot float

    # Add artificial floor item if placing on the bottom
    floor_item = None
    if abs(pos[Z]) < epsilon:
         # Create a dummy floor item covering the entire ULD base
         floor_item = Item(id=-100, size=(uld_bbox[X], uld_bbox[Y], 0), weight=0,
                           is_rotatable=False, is_tiltable=False, is_stackable=True)
         floor_item.position = (0, 0, 0)
         potential_supporters.add(floor_item)


    # Filter relevant supporters based on height (already partially done by grid accelerator)
    relevant_supporters = set()
    for supp in potential_supporters:
        if supp.position is None: continue
        supp_top_z = supp.position[Z] + supp.current_size[Z]
        # Check if supporter's top surface is within padding range below item's bottom
        if pos[Z] - max_padding_height - epsilon <= supp_top_z <= pos[Z] + epsilon:
             # Check for horizontal overlap
             overlap_x = (pos[X] < supp.position[X] + supp.current_size[X] - epsilon) and \
                         (supp.position[X] < pos[X] + item_to_place.current_size[X] - epsilon)
             overlap_y = (pos[Y] < supp.position[Y] + supp.current_size[Y] - epsilon) and \
                         (supp.position[Y] < pos[Y] + item_to_place.current_size[Y] - epsilon)
             if overlap_x and overlap_y:
                  relevant_supporters.add(supp)

    if floor_item and floor_item not in relevant_supporters and abs(pos[Z]) < epsilon :
         # If placing on floor Z=0, ensure floor is considered if not already added
         relevant_supporters.add(floor_item)


    # Sort supporters by Z end position (top surface height), descending (Algorithm 2, line 3)
    sorted_supporters = sorted(list(relevant_supporters), key=lambda s: s.position[Z] + s.current_size[Z] if s.position else -1, reverse=True)

    directly_supported = False
    total_supported_area = 0.0
    supported_corners = [False, False, False, False] # Bottom corners: (minX,minY), (maxX,minY), (minX,maxY), (maxX,maxY)
    item_corners = [
        (pos[X], pos[Y]),
        (pos[X] + item_to_place.current_size[X], pos[Y]),
        (pos[X], pos[Y] + item_to_place.current_size[Y]),
        (pos[X] + item_to_place.current_size[X], pos[Y] + item_to_place.current_size[Y]),
    ]

    processed_areas: List[Tuple[float, float, float, float]] = [] # Store (x_min, y_min, x_max, y_max) of supporting areas

    for idx, supp in enumerate(sorted_supporters):
        if supp.position is None: continue
        supp_top_z = supp.position[Z] + supp.current_size[Z]
        is_direct_support = abs(supp_top_z - pos[Z]) < epsilon

        # Stackability check (Algorithm 2, lines 8-13)
        if not supp.is_stackable:
            if is_direct_support:
                # Check if the new item *actually* rests on this non-stackable item
                overlap_x = (pos[X] < supp.position[X] + supp.current_size[X] - epsilon) and \
                            (supp.position[X] < pos[X] + item_to_place.current_size[X] - epsilon)
                overlap_y = (pos[Y] < supp.position[Y] + supp.current_size[Y] - epsilon) and \
                            (supp.position[Y] < pos[Y] + item_to_place.current_size[Y] - epsilon)
                if overlap_x and overlap_y:
                    # print(f"Debug: Stackability violation: Item {item_to_place.id} on non-stackable Item {supp.id}")
                    if floor_item: potential_supporters.remove(floor_item) # Clean up floor item
                    return False # Cannot place directly on non-stackable item
            continue # Non-stackable item cannot provide support (direct or indirect)

        # Calculate overlap area between item_to_place base and supporter top
        overlap_x_min = max(pos[X], supp.position[X])
        overlap_y_min = max(pos[Y], supp.position[Y])
        overlap_x_max = min(pos[X] + item_to_place.current_size[X], supp.position[X] + supp.current_size[X])
        overlap_y_max = min(pos[Y] + item_to_place.current_size[Y], supp.position[Y] + supp.current_size[Y])

        current_overlap_area = max(0, overlap_x_max - overlap_x_min) * max(0, overlap_y_max - overlap_y_min)

        if current_overlap_area > epsilon:
            if is_direct_support:
                directly_supported = True # Algorithm 2, line 16

            # --- Simplified Area Calculation (Addressing Section 4.4.3 Issue) ---
            # Instead of complex subtraction, we calculate the union of supporting areas projection.
            # Add the current supporting rectangle to a list.
            current_rect = (overlap_x_min, overlap_y_min, overlap_x_max, overlap_y_max)

            # Check corners support
            for i in range(4):
                 if not supported_corners[i]:
                      corner_x, corner_y = item_corners[i]
                      # Check if corner is within the current supporting rectangle
                      if (overlap_x_min - epsilon <= corner_x <= overlap_x_max + epsilon and
                          overlap_y_min - epsilon <= corner_y <= overlap_y_max + epsilon):
                           supported_corners[i] = True

            processed_areas.append(current_rect)


    # Calculate the total supported area by finding the area of the union of rectangles
    # This is a complex geometric problem. A common approach is using a sweep-line algorithm
    # or pixel-based approximation. Here, we use a simplified approximation: Sum of areas minus
    # pairwise overlaps (inclusion-exclusion principle, simplified for pairwise).
    # NOTE: This simplification is NOT the same as the PDF's simplification but aims to be more accurate
    #       while still avoiding the full union calculation complexity. It can still overestimate or underestimate.
    #       A more robust method (like interval-based or sweep-line) is recommended for production.

    total_supported_area = 0
    for i in range(len(processed_areas)):
        rect_i = processed_areas[i]
        area_i = (rect_i[2] - rect_i[0]) * (rect_i[3] - rect_i[1])
        overlap_with_others = 0
        # Subtract overlaps with *previous* rectangles to avoid double counting subtraction
        for j in range(i):
             rect_j = processed_areas[j]
             # Calculate intersection area between rect_i and rect_j
             inter_x_min = max(rect_i[0], rect_j[0])
             inter_y_min = max(rect_i[1], rect_j[1])
             inter_x_max = min(rect_i[2], rect_j[2])
             inter_y_max = min(rect_i[3], rect_j[3])
             intersection_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
             overlap_with_others += intersection_area

        total_supported_area += max(0, area_i - overlap_with_others) # Add the unique contribution of this area


    # Final check (Algorithm 2, line 26)
    num_supported_corners = sum(supported_corners)
    support_ratio = total_supported_area / item_base_area if item_base_area > 0 else 1.0

    # Condition (i) or (ii) from PDF
    # (i) direct support AND sufficient area overlap
    condition_i = directly_supported and support_ratio >= min_overlap_ratio - epsilon
    # (ii) four corners supported (implies direct support by definition of corners)
    condition_ii = num_supported_corners == 4

    is_valid = condition_i or condition_ii

    # if not is_valid:
    #     print(f"Debug: Floating check failed for Item {item_to_place.id} at {pos}. "
    #           f"DirectSupport={directly_supported}, SupportRatio={support_ratio:.2f}, Corners={num_supported_corners}")

    if floor_item: potential_supporters.remove(floor_item) # Clean up floor item
    return is_valid


def projection(start_point: Tuple[float, float, float], direction: int,
               loaded_items_and_dummies: List[Item], uld: ULD) -> Set[Tuple[float, float, float]]:
    """
    Performs projection to generate new extreme points along one axis.
    Implements Algorithm 4 from the PDF.

    Args:
        start_point (p): The starting point for the projection.
        direction (d): The dimension along which to project (0=X, 1=Y, 2=Z).
        loaded_items_and_dummies: List of all loaded items and dummy items.
        uld: The ULD object.

    Returns:
        A set of new extreme points generated by this projection.
    """
    new_extreme_points = set()
    current_projection_point = list(start_point) # Mutable copy
    non_proj_dims = [i for i in range(DIMENSIONS) if i != direction]
    theta, eta = non_proj_dims[0], non_proj_dims[1] # Non-projection dimensions
    blocking_items: Set[Item] = set() # B in Algorithm 4
    epsilon = 1e-6

    # Sort items by non-ascending end position in projection direction (line 5)
    sorted_items = sorted(
        loaded_items_and_dummies,
        key=lambda item: item.position[direction] + item.current_size[direction] if item.position else -float('inf'),
        reverse=True
    )

    projection_hit_direct = False

    for item_l in sorted_items:
        if item_l.position is None: continue
        pos_l = item_l.position
        size_l = item_l.current_size

        # Filter out items irrelevant for projection (lines 7-9)
        if pos_l[direction] >= start_point[direction] - epsilon: continue # Item starts at or beyond projection start
        if pos_l[theta] + size_l[theta] <= start_point[theta] + epsilon: continue # Item ends before projection ray in theta
        if pos_l[eta] + size_l[eta] <= start_point[eta] + epsilon: continue # Item ends before projection ray in eta

        # Filter out items whose projection is blocked by others (lines 10-12)
        is_blocked = False
        for item_b in blocking_items:
            if item_b.position is None: continue
            pos_b = item_b.position
            size_b = item_b.current_size
            # Check if item_l is "behind" item_b in the non-projection directions relative to start_point
            l_behind_b_theta = (pos_l[theta] >= pos_b[theta] + size_b[theta] - epsilon) or \
                               (start_point[theta] >= pos_b[theta] + size_b[theta] - epsilon)
            l_behind_b_eta = (pos_l[eta] >= pos_b[eta] + size_b[eta] - epsilon) or \
                             (start_point[eta] >= pos_b[eta] + size_b[eta] - epsilon)
            # Check if item_l is "in front of or aligned with" item_b in projection direction
            l_front_b_d = pos_l[direction] + size_l[direction] >= pos_b[direction] - epsilon

            # This blocking logic seems complex and might need refinement based on visual examples.
            # The PDF's pseudocode condition "(c_theta^l >= c_theta^b or p_theta >= c_theta^b)" seems unusual.
            # Let's try a simpler interpretation: item_l is blocked if its projection path
            # towards the origin is intersected by a blocking item's volume.
            # A simpler check: Is item_l's relevant surface farther from origin than item_b's surface?
            if pos_l[direction] + size_l[direction] <= pos_b[direction] + size_b[direction] + epsilon:
                 # Check overlap in non-projection dimensions
                 overlap_theta = max(0, min(start_point[theta], pos_l[theta] + size_l[theta]) - max(pos_b[theta], pos_l[theta]))
                 overlap_eta = max(0, min(start_point[eta], pos_l[eta] + size_l[eta]) - max(pos_b[eta], pos_l[eta]))
                 # Simplified blocking check: if item_l is behind or aligned with item_b in projection dir
                 # and overlaps significantly in the other two dimensions where item_b exists.
                 # This needs careful geometric validation. Let's assume the PDF logic for now, despite potential ambiguity.
                 # PDF check interpretation:
                 block_cond_theta = (pos_l[theta] >= pos_b[theta] + size_b[theta] - epsilon) or (start_point[theta] >= pos_b[theta] + size_b[theta] - epsilon)
                 block_cond_eta = (pos_l[eta] >= pos_b[eta] + size_b[eta] - epsilon) or (start_point[eta] >= pos_b[eta] + size_b[eta] - epsilon)

                 if block_cond_theta and block_cond_eta : # Approximation of PDF line 11
                      is_blocked = True
                      break # Blocked by item_b

        if is_blocked:
            continue

        # Add extreme point if valid (lines 13-19)
        # Condition: Projection must hit the item's surface facing away from origin
        # And item must be stackable if projection is in Z direction (d=2)
        can_support = (direction != Z) or item_l.is_stackable
        projection_hits_surface = pos_l[direction] + size_l[direction] <= start_point[direction] + epsilon

        if projection_hits_surface and can_support:
            new_ep = list(start_point)
            new_ep[direction] = pos_l[direction] + size_l[direction] # Project onto the item's surface
            new_extreme_points.add(tuple(new_ep))

            # Stop if projection directly hits the item (lines 18-19)
            # Direct hit: start_point ray intersects item_l volume before hitting surface
            direct_hit_theta = start_point[theta] >= pos_l[theta] - epsilon and start_point[theta] < pos_l[theta] + size_l[theta] + epsilon
            direct_hit_eta = start_point[eta] >= pos_l[eta] - epsilon and start_point[eta] < pos_l[eta] + size_l[eta] + epsilon

            if direct_hit_theta and direct_hit_eta:
                 projection_hit_direct = True
                 break # Stop projection along this line

        # Add item to list of potentially blocking items (line 21)
        # Only add if it could potentially block future projections starting from the same point
        if pos_l[theta] < start_point[theta] + epsilon and pos_l[eta] < start_point[eta] + epsilon:
             blocking_items.add(item_l)


    # Project to ULD wall if projection was not stopped by a direct hit (lines 22-23)
    if not projection_hit_direct:
        # TODO: Handle projection onto tilted ULD walls correctly.
        # This requires calculating intersection of the ray (start_point moving along -direction)
        # with the plane equations of the ULD facets.
        # Simplified: Project onto the coordinate plane (wall at 0).
        final_ep = list(start_point)
        final_ep[direction] = 0.0 # Project onto wall at origin
        new_extreme_points.add(tuple(final_ep))

    return new_extreme_points


def generate_new_extreme_points(item: Item, loaded_items_and_dummies: List[Item], uld: ULD) -> Set[Tuple[float, float, float]]:
    """
    Generates new extreme points after placing an item.
    Implements Algorithm 3 from the PDF.

    Args:
        item: The newly placed item.
        loaded_items_and_dummies: List of all items in the ULD (including dummies).
        uld: The ULD object.

    Returns:
        A set of new extreme points.
    """
    new_points = set()
    if item.position is None:
        return new_points

    pos = item.position
    size = item.current_size

    # Define projection starting points (modified from Crainic et al., see Sec 4.5, Fig 9b)
    start_points_projections = [
        # Start point for projections in X and Y direction
        ((pos[X] + size[X], pos[Y] + size[Y], pos[Z]), [X, Y]),
        # Start point for projections in X and Z direction
        ((pos[X], pos[Y] + size[Y], pos[Z] + size[Z]), [X, Z]),
        # Start point for projections in Y and Z direction
        ((pos[X] + size[X], pos[Y], pos[Z] + size[Z]), [Y, Z]),
    ]

    for start_point, directions in start_points_projections:
        for proj_dir in directions:
            # Skip Z-projection start points if item is non-stackable (lines 4-6)
            # This check needs careful interpretation. The PDF says "j=3" (meaning Z-axis for the start point offset)
            # and projection direction d != j.
            # Let's interpret: If the start point is on TOP of the item (uses pos[Z]+size[Z]),
            # and the item is non-stackable, don't project horizontally (X or Y) from there.
            is_start_on_top = abs(start_point[Z] - (pos[Z] + size[Z])) < 1e-6
            if not item.is_stackable and is_start_on_top and proj_dir != Z:
                 continue

            new_points.update(projection(start_point, proj_dir, loaded_items_and_dummies, uld))

    # Add extreme point on top of the item if stackable (lines 14-16)
    if item.is_stackable:
        new_points.add((pos[X], pos[Y], pos[Z] + size[Z]))

    return new_points


def insertion_heuristic(items_to_load: List[Item], uld: ULD, sort_criterion: str,
                        randomization_degree: float, use_substructure: bool,
                        min_overlap_ratio: float, max_padding_height: float,
                        max_cog_deviation: float, grid_accelerator: GridBasedAccelerator) -> ULD:
    """
    Attempts to load items into a single ULD using the extreme point heuristic.
    Implements Algorithm 1 from the PDF.

    Args:
        items_to_load: List of items available for loading.
        uld: The ULD to load into.
        sort_criterion: Sorting method for items.
        randomization_degree: Randomization factor for sorting (0 to 1).
        use_substructure: Whether to use a substructure for this run.
        min_overlap_ratio: Minimum support area ratio required.
        max_padding_height: Maximum padding allowed.
        max_cog_deviation: Max allowed CoG deviation from geometric center (as ratio of bbox dim).
        grid_accelerator: The grid accelerator instance for this ULD.

    Returns:
        The ULD object with loaded items.
    """
    # Reset ULD state for this run
    uld.loaded_items = []
    # Clear and rebuild grid
    grid_accelerator.grid = {}


    # 1. Initialize extreme points (line 1)
    extreme_points: Set[Tuple[float, float, float]] = {(0.0, 0.0, 0.0)}

    # 2. Adapt ULD (handle edge/substructure by adding dummy items) (line 2)
    adapt_uld(uld, use_substructure)
    all_items_in_uld = uld.dummy_items[:] # Start with dummy items
    # Add dummy items to the grid
    for dummy in uld.dummy_items:
         grid_accelerator.add_item(dummy)
         extreme_points.update(generate_new_extreme_points(dummy, all_items_in_uld, uld))


    # 3. Determine ordered list of items and orientations (lines 3)
    # The list contains tuples: (item, [list_of_orientations_for_this_group])
    ordered_item_list = group_and_sort_items(items_to_load, sort_criterion, randomization_degree)

    items_processed_in_run = set() # Track items loaded in this heuristic run

    # 4. Iterate through sorted items (line 4)
    for item, possible_orientations in ordered_item_list:
        if item.id in items_processed_in_run:
             continue # Item already loaded in this run

        item_loaded = False
        processed_eps_for_item = set() # Track EPs tried for this item to avoid infinite loops

        # Keep trying extreme points until item is loaded or no points left (while loop in line 5)
        while True:
            # Select next EP based on z, y, x sorting (Section 4.3)
            current_ep = get_next_extreme_point(extreme_points - processed_eps_for_item)

            if current_ep is None:
                # print(f"Debug: No more valid extreme points for Item {item.id}")
                break # No more candidate points for this item

            processed_eps_for_item.add(current_ep) # Mark EP as tried for this item

            # 5. Iterate through possible orientations for the current item/group (line 6)
            for orientation in possible_orientations:
                item.set_orientation(orientation)
                item_size = item.current_size

                # 6. Potentially move the extreme point (line 7 / Section 4.3)
                potential_pos = move_extreme_point(current_ep, item_size, uld)

                # 7. Check if item can be loaded at the (potentially moved) point (line 7)
                # Check all constraints: ULD fit, collision, non-floating, stackability
                # Use grid accelerator to get potential colliders/supporters
                potential_colliders = grid_accelerator.get_potential_colliders(item, potential_pos)
                potential_supporters = grid_accelerator.get_potential_supporters(item, potential_pos, max_padding_height)
                # Add dummy items to potential colliders/supporters as they affect placement
                potential_colliders.update(uld.dummy_items)
                potential_supporters.update(uld.dummy_items)


                fits_uld = check_uld_fit(item, potential_pos, uld)
                no_collision = check_collision(item, potential_pos, potential_colliders)
                is_supported = check_non_floating_and_stackability(item, potential_pos, potential_supporters, uld.bounding_box, min_overlap_ratio, max_padding_height)

                if fits_uld and no_collision and is_supported:
                    # Check weight capacity (simplified - check before placing)
                    if uld.get_total_weight() + item.weight > uld.weight_capacity:
                         continue # Skip if weight exceeds capacity

                    # --- Check CoG constraint (simplified - check before placing) ---
                    # Temporarily add item to calculate potential new CoG
                    uld.loaded_items.append(item)
                    item.position = potential_pos # Temporarily set position
                    new_cog = uld.calculate_center_of_gravity()
                    uld.loaded_items.pop() # Remove temporary item
                    item.position = None

                    cog_valid = True
                    if new_cog:
                         dev_x = abs(new_cog[X] - uld.geometric_center[X]) / uld.bounding_box[X] if uld.bounding_box[X] > 0 else 0
                         dev_y = abs(new_cog[Y] - uld.geometric_center[Y]) / uld.bounding_box[Y] if uld.bounding_box[Y] > 0 else 0
                         if dev_x > max_cog_deviation or dev_y > max_cog_deviation:
                              cog_valid = False
                              # print(f"Debug: CoG check failed for Item {item.id} at {potential_pos}. DevX={dev_x:.2f}, DevY={dev_y:.2f}")


                    # --- End CoG Check ---

                    if cog_valid:
                        # Load the item (line 8)
                        item.position = potential_pos
                        uld.loaded_items.append(item)
                        all_items_in_uld.append(item)
                        items_processed_in_run.add(item.id)
                        grid_accelerator.add_item(item) # Add to grid

                        # Remove the used extreme point (implicitly handled by not selecting it again)
                        # The original paper suggests removing EPs *inside* the new item.
                        # For simplicity here, we just remove the one used.
                        # A full implementation should remove points covered by the new item.
                        if current_ep in extreme_points:
                             extreme_points.remove(current_ep)

                        # Update set of extreme points (line 10 / Algorithm 3)
                        new_eps = generate_new_extreme_points(item, all_items_in_uld, uld)

                        # Filter new EPs: must be inside ULD bounds (basic check)
                        L, W, H = uld.bounding_box
                        filtered_new_eps = set()
                        for nep in new_eps:
                             if 0 <= nep[X] <= L + 1e-6 and \
                                0 <= nep[Y] <= W + 1e-6 and \
                                0 <= nep[Z] <= H + 1e-6:
                                 # Further check: EP should not be *inside* any existing item
                                 is_inside_item = False
                                 potential_containing_items = grid_accelerator.get_potential_colliders(Item(-999, (1,1,1),0,0,0,0), nep) # Check items around EP
                                 potential_containing_items.update(uld.dummy_items)
                                 for loaded_item in potential_containing_items:
                                      if loaded_item.position is None: continue
                                      p = loaded_item.position
                                      s = loaded_item.current_size
                                      if p[X] - 1e-6 < nep[X] < p[X] + s[X] + 1e-6 and \
                                         p[Y] - 1e-6 < nep[Y] < p[Y] + s[Y] + 1e-6 and \
                                         p[Z] - 1e-6 < nep[Z] < p[Z] + s[Z] + 1e-6:
                                          is_inside_item = True
                                          break
                                 if not is_inside_item:
                                      filtered_new_eps.add(nep)


                        extreme_points.update(filtered_new_eps)

                        item_loaded = True
                        # print(f"Loaded Item {item.id} at {potential_pos} with orientation {orientation}")
                        break # Go to the next item (line 11)

            if item_loaded:
                break # Break from orientation loop

        if item_loaded:
            continue # Continue to next item in ordered_item_list

        # If item couldn't be loaded after trying all EPs for it
        # print(f"Could not load Item {item.id}")


    # Remove dummy items from final loaded list
    uld.loaded_items = [item for item in uld.loaded_items if item.id >= 0]

    # Return the ULD with loaded items (line 16)
    return uld


# --- Example Usage ---
if __name__ == "__main__":
    # Define some sample items
    items = [
        Item(id=1, size=(10, 10, 10), weight=5, is_rotatable=True, is_tiltable=False, is_stackable=True),
        Item(id=2, size=(5, 15, 8), weight=8, is_rotatable=True, is_tiltable=False, is_stackable=True),
        Item(id=3, size=(12, 12, 12), weight=10, is_rotatable=False, is_tiltable=False, is_stackable=False),
        Item(id=4, size=(8, 8, 20), weight=12, is_rotatable=True, is_tiltable=False, is_stackable=True),
        Item(id=5, size=(20, 5, 5), weight=7, is_rotatable=True, is_tiltable=False, is_stackable=True),
    ]

    # Define a sample ULD (simple cuboid for this example)
    uld_dims = (30, 30, 30)
    uld_vertices = [ # Vertices of a cuboid at origin
        (0, 0, 0), (uld_dims[X], 0, 0), (uld_dims[X], uld_dims[Y], 0), (0, uld_dims[Y], 0),
        (0, 0, uld_dims[Z]), (uld_dims[X], 0, uld_dims[Z]), (uld_dims[X], uld_dims[Y], uld_dims[Z]), (0, uld_dims[Y], uld_dims[Z])
    ]
    # Simplified facets for cuboid
    uld_facets = [
        [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
        [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]
    ]
    sample_uld = ULD(id=101, vertices=uld_vertices, facets=uld_facets,
                     weight_capacity=100, volume_capacity=uld_dims[X]*uld_dims[Y]*uld_dims[Z],
                     edge_width=2, vertical_edge_offset=5, use_substructure_allowed=True) # Example edge params

    # Parameters from Table 2 (adapted)
    params = {
        "min_overlap_ratio": 0.9,
        "max_padding_height": 0, # Set to 0 as padding logic is complex/simplified
        "max_cog_deviation": 0.1, # 10% deviation allowed
        "randomization_degree": 0.0, # No randomization for single run test
        "use_substructure": False, # Test without substructure first
        "sort_criterion": 'stackability-cumulated_volume'
    }

    # Initialize Grid Accelerator
    grid_accel = GridBasedAccelerator(sample_uld.bounding_box, items)


    print("--- Running Insertion Heuristic ---")
    print(f"ULD before loading: {sample_uld}")
    print(f"Items to load: {len(items)}")

    # Run the heuristic
    loaded_uld = insertion_heuristic(
        items_to_load=items,
        uld=sample_uld,
        sort_criterion=params["sort_criterion"],
        randomization_degree=params["randomization_degree"],
        use_substructure=params["use_substructure"],
        min_overlap_ratio=params["min_overlap_ratio"],
        max_padding_height=params["max_padding_height"],
        max_cog_deviation=params["max_cog_deviation"],
        grid_accelerator=grid_accel
    )

    print("\n--- Loading Results ---")
    print(f"ULD after loading: {loaded_uld}")
    print("Loaded Items:")
    if not loaded_uld.loaded_items:
        print("  None")
    else:
        for item in loaded_uld.loaded_items:
            print(f"  - {item}")

    final_cog = loaded_uld.calculate_center_of_gravity()
    print(f"Final CoG (x, y): {final_cog}")
    print(f"Geometric Center (x, y): {loaded_uld.geometric_center}")
    if final_cog:
        dev_x = abs(final_cog[X] - loaded_uld.geometric_center[X]) / loaded_uld.bounding_box[X] if loaded_uld.bounding_box[X] > 0 else 0
        dev_y = abs(final_cog[Y] - loaded_uld.geometric_center[Y]) / loaded_uld.bounding_box[Y] if loaded_uld.bounding_box[Y] > 0 else 0
        print(f"CoG Deviation (x, y): ({dev_x*100:.1f}%, {dev_y*100:.1f}%)")

    # Items not loaded
    loaded_item_ids = {item.id for item in loaded_uld.loaded_items}
    not_loaded_items = [item for item in items if item.id not in loaded_item_ids]
    print("\nItems not loaded:")
    if not not_loaded_items:
        print("  All items loaded.")
    else:
        for item in not_loaded_items:
            print(f"  - Item {item.id}")

    # Example with substructure
    print("\n--- Running Insertion Heuristic with Substructure ---")
    params["use_substructure"] = True
    sample_uld_sub = ULD(id=102, vertices=uld_vertices, facets=uld_facets,
                         weight_capacity=100, volume_capacity=uld_dims[X]*uld_dims[Y]*uld_dims[Z],
                         edge_width=2, vertical_edge_offset=5, use_substructure_allowed=True)
    grid_accel_sub = GridBasedAccelerator(sample_uld_sub.bounding_box, items)
    loaded_uld_sub = insertion_heuristic(
        items_to_load=items, uld=sample_uld_sub, grid_accelerator=grid_accel_sub, **params
    )
    print(f"ULD after loading with substructure: {loaded_uld_sub}")
    print("Loaded Items:")
    if not loaded_uld_sub.loaded_items:
        print("  None")
    else:
        for item in loaded_uld_sub.loaded_items:
            print(f"  - {item}")


