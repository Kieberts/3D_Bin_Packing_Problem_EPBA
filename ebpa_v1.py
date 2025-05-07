import math
import random
import logging
from typing import List, Tuple, Set, Optional, Dict, Any, NamedTuple
from enum import Enum

# --- Konfiguration und Konstanten ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
WIDTH_INDEX, HEIGHT_INDEX, DEPTH_INDEX = 0, 1, 2
DIMENSIONS = 3
GRAMS_TO_KG = 1000.0
EPSILON = 1e-6

class Orientation(Enum):
    WHD = 0
    DHW = 1

class Point(NamedTuple):
    x: float
    y: float
    z: float

class Size(NamedTuple):
    width: float
    height: float
    depth: float

# --- Kernklassen (Product, Container) ---
class Product:
    def __init__(self, id: str, size: Tuple[float, float, float], weight_grams: float, upright_only: bool):
        self.id = id
        self.original_size = Size(max(0.0, size[WIDTH_INDEX]),
                                  max(0.0, size[HEIGHT_INDEX]),
                                  max(0.0, size[DEPTH_INDEX]))
        self.weight_kg = max(0.0, weight_grams / GRAMS_TO_KG)
        self.upright_only = upright_only
        self._current_orientation: Orientation = Orientation.WHD
        self.position: Optional[Point] = None
        self.current_size = self._calculate_current_size()

    def set_orientation(self, orientation: Orientation):
        if self.upright_only and orientation != Orientation.WHD:
            self._current_orientation = Orientation.WHD
        else:
            self._current_orientation = orientation
        self.current_size = self._calculate_current_size()


    def _calculate_current_size(self) -> Size:
        w, h, d = self.original_size
        if self._current_orientation == Orientation.DHW:
            return Size(d, h, w)
        else:
            return Size(w, h, d)

    def get_possible_orientations(self) -> List[Orientation]:
        if self.upright_only:
            return [Orientation.WHD]
        else:
            w, _, d = self.original_size
            if abs(w - d) < EPSILON or w <= EPSILON or d <= EPSILON:
                return [Orientation.WHD]
            else:
                return [Orientation.WHD, Orientation.DHW]

    @property
    def volume(self) -> float:
        return self.original_size.width * self.original_size.height * self.original_size.depth

    @property
    def end_position(self) -> Optional[Point]:
        if self.position is None:
            return None
        return Point(self.position.x + self.current_size.width,
                     self.position.y + self.current_size.height,
                     self.position.z + self.current_size.depth)

    def reset_placement(self):
        self.position = None
        self.set_orientation(Orientation.WHD)

    def __repr__(self):
        pos_str = f"({self.position.x:.1f},{self.position.y:.1f},{self.position.z:.1f})" if self.position else "None"
        size_str = f"({self.current_size.width:.1f},{self.current_size.height:.1f},{self.current_size.depth:.1f})"
        return (f"Product(id='{self.id}', size={size_str}, weight={self.weight_kg:.3f}kg, "
                f"orient={self._current_orientation.name}, pos={pos_str})")


class Container:
    def __init__(self, definition: Dict[str, Any]):
        self.name = definition.get("name", "Unnamed Container")
        self.dimensions = Size(max(0.0, definition.get("W", 0.0)),
                               max(0.0, definition.get("H", 0.0)),
                               max(0.0, definition.get("D", 0.0)))
        self.max_weight_kg = max(0.0, definition.get("max_weight", 0.0) / GRAMS_TO_KG)
        self._volume = self.dimensions.width * self.dimensions.height * self.dimensions.depth
        self.loaded_products: List[Product] = []

    @property
    def volume(self) -> float:
        return self._volume

    @property
    def total_weight_kg(self) -> float:
        return sum(product.weight_kg for product in self.loaded_products)

    @property
    def utilized_volume(self) -> float:
        return sum(prod.volume for prod in self.loaded_products)

    @property
    def utilization_ratio(self) -> float:
        if self.volume <= EPSILON:
            return 0.0
        return min(1.0, self.utilized_volume / self.volume)

    def reset(self):
        self.loaded_products = []

    def __repr__(self):
        dims_str = f"({self.dimensions.width:.1f},{self.dimensions.height:.1f},{self.dimensions.depth:.1f})"
        return (f"Container(name='{self.name}', dim={dims_str}, "
                f"products={len(self.loaded_products)}, weight={self.total_weight_kg:.3f}/{self.max_weight_kg:.3f}kg, "
                f"utilization={self.utilization_ratio * 100:.1f}%)")


# --- Beschleunigungsstruktur (GridBasedAccelerator) ---
class GridBasedAccelerator:
    def __init__(self, container_dims: Size, products: List[Product], epsilon: float = EPSILON):
        self.container_dims = container_dims
        self.epsilon = epsilon
        self.cell_size = self._calculate_cell_size(products)

        if self.cell_size <= self.epsilon:
             positive_dims = [d for d in container_dims if d > self.epsilon]
             self.cell_size = min(positive_dims) if positive_dims else 1.0
             if self.cell_size <= self.epsilon:
                  self.cell_size = 1.0
             logging.warning(f"Grid cell size was too small, adjusted to {self.cell_size:.2f}")

        self.grid_dimensions = (
            max(1, math.ceil(container_dims.width / self.cell_size)),
            max(1, math.ceil(container_dims.height / self.cell_size)),
            max(1, math.ceil(container_dims.depth / self.cell_size)),
        )
        self.grid: Dict[Tuple[int, int, int], Set[Product]] = {}

    def _calculate_cell_size(self, products: List[Product]) -> float:
        positive_dims = [d for d in self.container_dims if d > self.epsilon]
        min_container_dim = min(positive_dims) if positive_dims else 1.0

        valid_products = [p for p in products if p.volume > self.epsilon]
        if not valid_products:
            return max(self.epsilon * 2, min_container_dim)

        total_edge_sum = 0
        count = 0
        for prod in valid_products:
            if all(d > self.epsilon for d in prod.original_size):
                total_edge_sum += sum(prod.original_size)
                count += 3

        if count == 0:
            return max(self.epsilon * 2, min_container_dim)

        avg_size = total_edge_sum / count
        calculated_size = max(1.0, min(avg_size, min_container_dim))
        return max(self.epsilon * 2, calculated_size)

    def _get_grid_indices_for_point(self, point: Point) -> Tuple[int, int, int]:
        if self.cell_size <= self.epsilon: return (0, 0, 0)

        ix = max(0, min(math.floor((point.x + self.epsilon) / self.cell_size), self.grid_dimensions[WIDTH_INDEX] - 1))
        iy = max(0, min(math.floor((point.y + self.epsilon) / self.cell_size), self.grid_dimensions[HEIGHT_INDEX] - 1))
        iz = max(0, min(math.floor((point.z + self.epsilon) / self.cell_size), self.grid_dimensions[DEPTH_INDEX] - 1))
        return ix, iy, iz

    def _get_intersecting_cells_indices(self, position: Point, size: Size) -> Set[Tuple[int, int, int]]:
        indices = set()
        if self.cell_size <= self.epsilon or size.width <= 0 or size.height <= 0 or size.depth <= 0:
            return indices

        min_pt = position
        max_pt = Point(position.x + size.width - self.epsilon,
                       position.y + size.height - self.epsilon,
                       position.z + size.depth - self.epsilon)

        min_ix, min_iy, min_iz = self._get_grid_indices_for_point(min_pt)
        max_ix, max_iy, max_iz = self._get_grid_indices_for_point(max_pt)

        max_ix = max(min_ix, max_ix)
        max_iy = max(min_iy, max_iy)
        max_iz = max(min_iz, max_iz)

        for ix in range(min_ix, max_ix + 1):
            for iy in range(min_iy, max_iy + 1):
                for iz in range(min_iz, max_iz + 1):
                    if 0 <= ix < self.grid_dimensions[WIDTH_INDEX] and \
                       0 <= iy < self.grid_dimensions[HEIGHT_INDEX] and \
                       0 <= iz < self.grid_dimensions[DEPTH_INDEX]:
                        indices.add((ix, iy, iz))
        return indices

    def add_product(self, product: Product):
        if product.position is None: return
        indices = self._get_intersecting_cells_indices(product.position, product.current_size)
        for index_tuple in indices:
            self.grid.setdefault(index_tuple, set()).add(product)

    def remove_product(self, product: Product):
         if product.position is None: return
         indices = self._get_intersecting_cells_indices(product.position, product.current_size)
         for index_tuple in indices:
             if index_tuple in self.grid:
                 self.grid[index_tuple].discard(product)
                 if not self.grid[index_tuple]:
                      del self.grid[index_tuple]

    def get_potential_colliders(self, product_to_place: Product, position: Point) -> Set[Product]:
        potential_colliders = set()
        indices = self._get_intersecting_cells_indices(position, product_to_place.current_size)
        for index_tuple in indices:
            potential_colliders.update(self.grid.get(index_tuple, set()))
        potential_colliders.discard(product_to_place)
        return potential_colliders

    def get_potential_supporters(self, product_to_place: Product, position: Point) -> Set[Product]:
        potential_supporters = set()
        search_y = max(0.0, position.y - self.epsilon)
        support_pos = Point(position.x, search_y, position.z)
        support_size = Size(product_to_place.current_size.width, self.epsilon, product_to_place.current_size.depth)

        indices = self._get_intersecting_cells_indices(support_pos, support_size)
        candidates = set()
        for index_tuple in indices:
            candidates.update(self.grid.get(index_tuple, set()))
        candidates.discard(product_to_place)

        for prod in candidates:
            if prod.position is None: continue
            prod_end_pos = prod.end_position
            if prod_end_pos is None: continue

            if abs(prod_end_pos.y - position.y) < self.epsilon:
                if EpbaBinPacker._check_aabb_overlap(
                    position1=Point(position.x, 0, position.z),
                    size1=Size(product_to_place.current_size.width, 0, product_to_place.current_size.depth),
                    position2=Point(prod.position.x, 0, prod.position.z),
                    size2=Size(prod.current_size.width, 0, prod.current_size.depth),
                    check_y=False,
                    epsilon=self.epsilon):
                    potential_supporters.add(prod)
        return potential_supporters


# --- Hauptalgorithmus Klasse (EpbaBinPacker) ---
class EpbaBinPacker:
    def __init__(self, container_definitions: List[Dict[str, Any]], epsilon: float = EPSILON):
        self.available_containers = sorted(
            [Container(c) for c in container_definitions],
            key=lambda c: (c.volume, c.max_weight_kg)
        )
        self.epsilon = epsilon
        logging.info(f"Initialized EpbaBinPacker with {len(self.available_containers)} container types.")

    def pack(self, product_definitions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not product_definitions:
            logging.warning("No products provided to pack.")
            return None

        products_to_pack = self._prepare_products(product_definitions)
        num_products_to_pack = len(products_to_pack)
        if num_products_to_pack == 0:
             logging.warning("No valid products found in definitions.")
             return None

        logging.info(f"Attempting to pack {num_products_to_pack} products.")

        total_product_volume = sum(p.volume for p in products_to_pack)
        total_product_weight_kg = sum(p.weight_kg for p in products_to_pack)

        # Iteriere durch die Container und versuche zu packen
        for container in self.available_containers:
            logging.debug(f"Checking container: {container.name}")
            # Vorabprüfungen für diesen Container
            if not self._passes_pre_checks(container, products_to_pack, total_product_volume, total_product_weight_kg):
                continue # Nächsten Container prüfen

            # Versuche, in diesen Container zu packen
            logging.info(f"--- Attempting to pack in Container: {container.name} ---")
            success = self._try_pack_in_container(container, products_to_pack)

            if success:
                logging.info(f"--- Successfully packed all {num_products_to_pack} products in Container: {container.name} ---")
                return self._format_result(container)
            else:
                # Logging für Fehlschlag erfolgt in _try_pack_in_container
                logging.info(f"--- Failed to pack all {num_products_to_pack} products in Container: {container.name} ---")
                # Wichtig: Produkte und Container für den nächsten Versuch zurücksetzen
                container.reset()
                for p in products_to_pack:
                    p.reset_placement()

        # Wenn die Schleife endet, wurde kein passender Container gefunden
        logging.warning("--- No suitable container found for all products. ---")
        return None

    def _prepare_products(self, product_definitions: List[Dict[str, Any]]) -> List[Product]:
        products = []
        for i, p_def in enumerate(product_definitions):
             try:
                  w = p_def.get("w", 0.0)
                  h = p_def.get("h", 0.0)
                  d = p_def.get("d", 0.0)
                  weight_g = p_def.get("weight", 0.0)
                  product_id = p_def.get("id", f"product_{i}")
                  upright = p_def.get("upright_only", False)

                  # Erstelle Produkt nur, wenn Dimensionen > 0 sind (Gewicht kann 0 sein)
                  if w > 0 and h > 0 and d > 0:
                        product = Product(product_id, (w, h, d), weight_g, upright)
                        products.append(product)
                  else:
                       logging.warning(f"Skipping product definition due to invalid dimensions: {p_def}")
             except Exception as e:
                  logging.error(f"Error creating product from definition {p_def}: {e}")
        return products

    def _passes_pre_checks(self, container: Container, products: List[Product], total_volume: float, total_weight: float) -> bool:
        """Führt die Vorabprüfungen für einen gegebenen Container durch."""
        container.reset() # Stelle sicher, dass der Container leer ist für die Prüfung

        # 1. Grundlegende Dimensionsprüfung
        if container.dimensions.width <= 0 or container.dimensions.height <= 0 or container.dimensions.depth <= 0:
            logging.debug(f"Skipping container {container.name}: Invalid dimensions.")
            return False

        # 2. Gesamtvolumenprüfung
        if total_volume > container.volume + self.epsilon:
            logging.debug(f"Skipping container {container.name}: Total product volume ({total_volume:.2f}) exceeds container volume ({container.volume:.2f}).")
            return False

        # 3. Gesamtgewichtprüfung (intern in kg)
        if total_weight > container.max_weight_kg + self.epsilon:
            logging.debug(f"Skipping container {container.name}: Total product weight ({total_weight:.3f}kg) exceeds max weight ({container.max_weight_kg:.3f}kg).")
            return False

        # 4. Individuelle Produkt-Passform-Prüfung
        if not self._check_all_products_can_potentially_fit(products, container):
            logging.debug(f"Skipping container {container.name}: Not all products can individually fit.")
            return False

        # Alle Vorabprüfungen bestanden
        return True

    def _check_all_products_can_potentially_fit(self, products: List[Product], container: Container) -> bool:
        """ Prüft, ob jedes einzelne Produkt theoretisch in den Container passt (in irgendeiner Orientierung). """
        cont_dims_sorted = sorted(container.dimensions)
        for product in products:
            if product.volume <= self.epsilon: continue

            can_fit_this_product = False
            original_orientation = product._current_orientation # Orientierung merken
            for orientation in product.get_possible_orientations():
                product.set_orientation(orientation)
                prod_dims_sorted = sorted(product.current_size)
                if all(prod_dims_sorted[i] <= cont_dims_sorted[i] + self.epsilon for i in range(DIMENSIONS)):
                    can_fit_this_product = True
                    break

            product.set_orientation(original_orientation) # Orientierung zurücksetzen

            if not can_fit_this_product:
                return False
        return True

    def _try_pack_in_container(self, container: Container, products_to_pack: List[Product]) -> bool:
        """ Versucht, die Produkte mittels der Heuristik in den gegebenen Container zu packen. """
        container.reset() # Sicherstellen, dass der Container leer startet
        for p in products_to_pack:
            p.reset_placement() # Produktzustand zurücksetzen

        valid_products_for_grid = [p for p in products_to_pack if p.volume > self.epsilon]
        if not valid_products_for_grid and products_to_pack:
             logging.warning("Packing attempted with only zero-volume products.")
             return True
        elif not valid_products_for_grid:
             return True # Keine Produkte -> erfolgreich

        try:
            grid_accel = GridBasedAccelerator(container.dimensions, valid_products_for_grid, self.epsilon)
        except Exception as e:
            logging.error(f"Error initializing GridAccelerator for {container.name}: {e}")
            return False

        sort_criterion = 'weight_desc' # Standard: Schwerste zuerst
        ordered_products = self._sort_products(products_to_pack, sort_criterion)

        extreme_points: Set[Point] = {Point(0.0, 0.0, 0.0)}

        for i, product in enumerate(ordered_products):
            best_placement = self._find_best_placement_for_product(
                product, extreme_points, container, grid_accel
            )

            if best_placement:
                position, orientation = best_placement
                product.set_orientation(orientation)
                product.position = position
                container.loaded_products.append(product)
                grid_accel.add_product(product)
                extreme_points = self._update_extreme_points(product, extreme_points, container)
            else:
                logging.warning(f"Could not place product {product.id} (index {i}, size {product.original_size}) in container {container.name}. Aborting attempt for this container.")
                return False # Packen für diesen Container fehlgeschlagen

        return len(container.loaded_products) == len(products_to_pack)

    def _format_result(self, container: Container) -> Dict[str, Any]:
        placements = []
        for prod in container.loaded_products:
            pos = prod.position if prod.position is not None else Point(0.0, 0.0, 0.0)
            size = prod.current_size
            placements.append({
                "id": prod.id,
                "x": pos.x, "y": pos.y, "z": pos.z,
                "w": size.width, "h": size.height, "d": size.depth,
                "orientation": prod._current_orientation.name,
                "weight": prod.weight_kg * GRAMS_TO_KG
            })
        result_dict = {
            "container": {
                "name": container.name,
                "width": container.dimensions.width,
                "height": container.dimensions.height,
                "depth": container.dimensions.depth,
                "max_weight": container.max_weight_kg * GRAMS_TO_KG,
                "utilization": container.utilization_ratio
            },
            "placements": placements
        }
        return result_dict

    def _sort_products(self, products: List[Product], criterion: str) -> List[Product]:
        key_func: Any 
        if criterion == 'volume_desc':
            key_func = lambda p: (-p.volume, -p.original_size.height)
        elif criterion == 'weight_desc':
            key_func = lambda p: (-p.weight_kg, -p.volume)
        elif criterion == 'random':
             shuffled = products[:]
             random.shuffle(shuffled)
             return shuffled
        else:
            logging.warning(f"Unknown sort criterion '{criterion}', defaulting to 'weight_desc'.")
            key_func = lambda p: (-p.weight_kg, -p.volume)
        return sorted(products, key=key_func)

    @staticmethod
    def _get_next_extreme_point(extreme_points: Set[Point]) -> Optional[Point]:
        if not extreme_points:
            return None
        return min(extreme_points, key=lambda p: (p.y, p.z, p.x))

    def _find_best_placement_for_product(self, product: Product, current_extreme_points: Set[Point],
                                         container: Container, grid_accel: GridBasedAccelerator
                                         ) -> Optional[Tuple[Point, Orientation]]:
        processed_eps_for_product: Set[Point] = set()
        # Iteriere durch sortierte Liste, um konsistente Reihenfolge zu haben
        sorted_ep_list = sorted(list(current_extreme_points), key=lambda p: (p.y, p.z, p.x))

        for current_ep in sorted_ep_list:
            if current_ep not in current_extreme_points or current_ep in processed_eps_for_product:
                continue

            original_orientation = product._current_orientation # Merken für Reset
            for orientation in product.get_possible_orientations():
                product.set_orientation(orientation)
                potential_pos = current_ep

                # --- Platzierungsprüfungen ---
                # 1. Passt es in die Containergrenzen?
                if not self._check_container_fit(product, potential_pos, container.dimensions):
                    continue

                # 2. Kollidiert es mit anderen?
                potential_colliders = grid_accel.get_potential_colliders(product, potential_pos)
                if not self._check_collision(product, potential_pos, potential_colliders):
                    continue

                # 3. Wird es gestützt & Gewicht okay?
                potential_supporters = grid_accel.get_potential_supporters(product, potential_pos)
                if not self._check_support_and_weight(product, potential_pos, potential_supporters):
                     continue

                # 4. Passt das Gesamtgewicht?
                if container.total_weight_kg + product.weight_kg > container.max_weight_kg + self.epsilon:
                    continue

                # --- Gültige Platzierung gefunden ---
                return potential_pos, orientation

            # Reset orientation if no valid placement found for this EP
            product.set_orientation(original_orientation)
            processed_eps_for_product.add(current_ep)

        # Keine gültige Platzierung gefunden
        return None


    def _check_container_fit(self, product: Product, pos: Point, container_dims: Size) -> bool:
        """ Prüft, ob das Produkt an der Position vollständig innerhalb der Containergrenzen liegt. """
        # Prüfe erst, ob das Produkt gültige Dimensionen hat
        if product.current_size.width <= 0 or product.current_size.height <= 0 or product.current_size.depth <= 0:
             return False # Produkt ohne Dimensionen kann nicht passen

        # Berechne die Endposition BASIEREND AUF DER POTENZIELLEN POSITION 'pos'
        prod_end_x = pos.x + product.current_size.width
        prod_end_y = pos.y + product.current_size.height
        prod_end_z = pos.z + product.current_size.depth

        # Prüfe, ob Start- und Endpunkte innerhalb der Grenzen liegen (mit Toleranz)
        if pos.x < -self.epsilon or pos.y < -self.epsilon or pos.z < -self.epsilon or \
           prod_end_x > container_dims.width + self.epsilon or \
           prod_end_y > container_dims.height + self.epsilon or \
           prod_end_z > container_dims.depth + self.epsilon:
            return False


        return True
    

    @staticmethod
    def _check_aabb_overlap(position1: Point, size1: Size, position2: Point, size2: Size,
                            check_x: bool = True, check_y: bool = True, check_z: bool = True,
                            epsilon: float = EPSILON) -> bool:
        overlap_x = True
        if check_x:
            overlap_x = (position1.x < position2.x + size2.width - epsilon) and \
                        (position2.x < position1.x + size1.width - epsilon)
        overlap_y = True
        if check_y:
            overlap_y = (position1.y < position2.y + size2.height - epsilon) and \
                        (position2.y < position1.y + size1.height - epsilon)
        overlap_z = True
        if check_z:
             overlap_z = (position1.z < position2.z + size2.depth - epsilon) and \
                         (position2.z < position1.z + size1.depth - epsilon)
        return overlap_x and overlap_y and overlap_z

    def _check_collision(self, product_to_place: Product, pos: Point,
                         potential_colliders: Set[Product]) -> bool:
        for loaded_prod in potential_colliders:
            if loaded_prod.position is None: continue
            if self._check_aabb_overlap(
                position1=pos, size1=product_to_place.current_size,
                position2=loaded_prod.position, size2=loaded_prod.current_size,
                epsilon=self.epsilon):
                
                return False
        return True

    def _check_support_and_weight(self, product_to_place: Product, pos: Point,
                                  potential_supporters: Set[Product]) -> bool:
        if abs(pos.y) < self.epsilon:
            return True # Am Boden unterstützt

        weight_constraint_violated = False
        is_supported = False

        for supporter in potential_supporters:
            if supporter.position is None: continue
            # Prüfung auf Überlappung etc. ist bereits in get_potential_supporters erfolgt
            is_supported = True # Ein potenzieller Supporter reicht

            # Strikte Gewichtsregel prüfen
            if product_to_place.weight_kg > supporter.weight_kg + self.epsilon:
                weight_constraint_violated = True
                break

        if weight_constraint_violated: return False
        if not is_supported:
            return False
        return True

    def _update_extreme_points(self, placed_product: Product, current_eps: Set[Point],
                              container: Container) -> Set[Point]:
        """Aktualisiert die Extrempunkte nach Platzierung (vereinfacht, ohne Grid-Nutzung hier)."""
        if placed_product.position is None: return current_eps

        pos = placed_product.position
        size = placed_product.current_size
        new_extreme_points = set(current_eps)

        # 1. Entferne verdeckte Punkte
        eps_to_remove = {
            ep for ep in new_extreme_points
            if (pos.x - self.epsilon <= ep.x < pos.x + size.width - self.epsilon and
                pos.y - self.epsilon <= ep.y < pos.y + size.height - self.epsilon and
                pos.z - self.epsilon <= ep.z < pos.z + size.depth - self.epsilon)
        }
        new_extreme_points.difference_update(eps_to_remove)

        # 2. Generiere neue potenzielle Punkte
        potential_new_eps: Set[Point] = set()
        if size.width > self.epsilon: potential_new_eps.add(Point(pos.x + size.width, pos.y, pos.z))
        if size.height > self.epsilon: potential_new_eps.add(Point(pos.x, pos.y + size.height, pos.z))
        if size.depth > self.epsilon: potential_new_eps.add(Point(pos.x, pos.y, pos.z + size.depth))

        # 3. Filtere neue Punkte
        final_new_eps: Set[Point] = set()
        cont_dims = container.dimensions
        for nep in potential_new_eps:
             # A. Innerhalb Container?
             if not (-self.epsilon <= nep.x <= cont_dims.width + self.epsilon and
                     -self.epsilon <= nep.y <= cont_dims.height + self.epsilon and
                     -self.epsilon <= nep.z <= cont_dims.depth + self.epsilon):
                 continue
             non_negative_nep = Point(max(0.0, nep.x), max(0.0, nep.y), max(0.0, nep.z))

             # B. Innerhalb anderer Produkte?
             if self._is_point_inside_any_product(non_negative_nep, container.loaded_products):
                  continue

             # C. Duplikat?
             is_duplicate = any(
                 abs(non_negative_nep.x - existing_ep.x) < self.epsilon and
                 abs(non_negative_nep.y - existing_ep.y) < self.epsilon and
                 abs(non_negative_nep.z - existing_ep.z) < self.epsilon
                 for existing_ep in new_extreme_points.union(final_new_eps)
             )
             if not is_duplicate:
                 final_new_eps.add(non_negative_nep)

        new_extreme_points.update(final_new_eps)
        return new_extreme_points

    def _is_point_inside_any_product(self, point: Point, loaded_products: List[Product]) -> bool:
        """ Prüft, ob ein Punkt *streng* innerhalb eines der geladenen Produkte liegt. """
        for prod in loaded_products:
            if prod.position is None: continue
            pos = prod.position; size = prod.current_size
            if (pos.x + self.epsilon < point.x < pos.x + size.width - self.epsilon and
                pos.y + self.epsilon < point.y < pos.y + size.height - self.epsilon and
                pos.z + self.epsilon < point.z < pos.z + size.depth - self.epsilon):
                return True
        return False

