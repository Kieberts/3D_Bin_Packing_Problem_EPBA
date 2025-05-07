from ebpa_v1 import EpbaBinPacker
from sim_class_v1 import BoxPlotter
import json


# # --- Example Usage ---
if __name__ == "__main__":
    # Container definitions
    containers = [
        {"name": "S", "W": 20, "H": 20, "D": 15, "max_weight": 3000, "volume": 6000},
        {"name": "M", "W": 25, "H": 25, "D": 20, "max_weight": 4000, "volume": 12500},
        {"name": "L", "W": 30, "H": 30, "D": 25, "max_weight": 6000, "volume": 22500},
        {"name": "XL", "W": 35, "H": 35, "D": 30, "max_weight": 8000, "volume": 36750},
        {"name": "XXL", "W": 40, "H": 40, "D": 35, "max_weight": 10000, "volume": 56000},
    ]

    # Product definitions
    # products_def = [
    #     {"id": "Smartphone", "w": 15, "h": 1, "d": 7, "weight": 0.2, "upright_only": False}, # Note: Height likely small
    #     {"id": "Watch", "w": 8, "h": 3, "d": 8, "weight": 0.1, "upright_only": False},
    #     {"id": "Camera", "w": 12, "h": 8, "d": 7, "weight": 0.5, "upright_only": False},
    #     {"id": "Headphones", "w": 20, "h": 15, "d": 10, "weight": 0.3, "upright_only": False},
    #     {"id": "Book", "w": 14, "h": 21, "d": 3, "weight": 0.6, "upright_only": False},
    #     {"id": "Mug", "w": 9, "h": 10, "d": 9, "weight": 0.4, "upright_only": True}, # Example upright
    #     # # Add a heavier item to test weight constraint
    #     {"id": "SmallWeight", "w": 5, "h": 5, "d": 5, "weight": 1.0, "upright_only": False},
    # ]

    products = [
        {"id": "Mini Drone1", "w": 25, "h": 10, "d": 25, "weight": 900, "upright_only": False},          
        {"id": "Mini Drone2", "w": 25, "h": 10, "d": 25, "weight": 900, "upright_only": False},          
        {"id": "Mini Drone3", "w": 25, "h": 10, "d": 25, "weight": 900, "upright_only": False},          
        {"id": "Mini Drone4", "w": 25, "h": 10, "d": 25, "weight": 900, "upright_only": False},          
        {"id": "Mini Drone5", "w": 25, "h": 10, "d": 25, "weight": 900, "upright_only": False},          
        {"id": "Mini Drone6", "w": 25, "h": 10, "d": 25, "weight": 900, "upright_only": False},          
    ]
    

    # Initialize packer
    packer = EpbaBinPacker(containers)

    # Run packing
    result = packer.pack(products)

    # Print result
    print("\n========= FINAL PACKING RESULT =========")
    if result:
        # # print(f"Successfully packed in Container: {json.dumps(result, indent=2)}")
        # print(f"Successfully packed in Container: {result['container_name']}")
        # print(f"Container Dimensions (W, H, D): {result['container_dimensions']}")
        # print(f"Total Weight: {result['total_weight']:.2f} kg")
        # print(f"Volume Utilization: {result['utilization_ratio']*100:.1f}%")
        # print("Packed Products Layout:")
        # for prod_info in result['packed_products']:
        #     pos = prod_info['position']
        #     size = prod_info['size_used']
        #     print(f"  - ID: {prod_info['id']}, Pos(x,y,z): ({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}), "
        #           f"Orient: {prod_info['orientation']}, Size(w',h',d'): {size}")
        print(json.dumps(result, indent=2))

        # Visualization
        plotter = BoxPlotter()
        plotter.plot(result)

    else:
        print("Packing failed: Could not fit all products into any available container.")
    print("========================================")

