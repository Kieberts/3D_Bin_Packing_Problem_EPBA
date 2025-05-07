from ebpa_v1 import EpbaBinPacker
from sim_class_v1 import BoxPlotter
import datetime
import os
# import json

def start_test(packer, pack_iterations, products, show_plot=False): 
    plannedIterations = "plannedIterations"
    noBin = "amountOfIterations"
    real_loop_count = 0
    counts = {plannedIterations: pack_iterations, noBin: 0, "S": 0, "M": 0, "L": 0, "XL": 0, "XXL": 0}
    for idx in range(pack_iterations):
        if idx % 10 == 0:
            print(f"Iteration: {idx} of {pack_iterations}")

        iteration_result = packer.pack(products)
        real_loop_count += 1

        # print(json.dumps(iteration_result, indent=2))
        # Visualization
        if show_plot:
            plotter = BoxPlotter()
            plotter.plot(iteration_result)

        container_name = iteration_result["container"].get("name")
        if container_name in counts:
            counts[container_name] += 1
    
    counts[noBin] = real_loop_count
    print("Test finished")

    return counts

def _test_data(amountOfProducts):
    containers = [
        {"name": "S", "W": 20, "H": 20, "D": 15, "max_weight": 3000, "volume": 6000},
        {"name": "M", "W": 25, "H": 25, "D": 20, "max_weight": 4000, "volume": 12500},
        {"name": "L", "W": 30, "H": 30, "D": 25, "max_weight": 6000, "volume": 22500},
        {"name": "XL", "W": 35, "H": 35, "D": 30, "max_weight": 8000, "volume": 36750},
        {"name": "XXL", "W": 40, "H": 40, "D": 35, "max_weight": 10000, "volume": 56000},
    ]
    products = []

    if amountOfProducts == 1:
        products = [
            {"id": "Smartphone", "w": 15, "h": 7, "d": 1, "weight": 200, "upright_only": False},
        ]      
    elif amountOfProducts == 2:
        products = [
            {"id": "Smartphone", "w": 15, "h": 7, "d": 1, "weight": 200, "upright_only": False},
            {"id": "Watch", "w": 8, "h": 8, "d": 3, "weight": 100, "upright_only": False},
        ]
    elif amountOfProducts == 3: 
        products = [
            {"id": "Smartphone", "w": 15, "h": 7, "d": 1, "weight": 200, "upright_only": False},
            {"id": "Watch", "w": 8, "h": 8, "d": 3, "weight": 100, "upright_only": False},
            {"id": "Digital Camera", "w": 12, "h": 8, "d": 7, "weight": 500, "upright_only": False},
        ]
    elif amountOfProducts == 4:
        products = [
            {"id": "Smartphone", "w": 15, "h": 7, "d": 1, "weight": 200, "upright_only": False},
            {"id": "Watch", "w": 8, "h": 8, "d": 3, "weight": 100, "upright_only": False},
            {"id": "Digital Camera", "w": 12, "h": 8, "d": 7, "weight": 500, "upright_only": False},
            {"id": "Headphone1", "w": 20, "h": 15, "d": 10, "weight": 300, "upright_only": False},
        ]
    elif amountOfProducts == 5:
        products = [
            {"id": "Smartphone", "w": 15, "h": 7, "d": 1, "weight": 200, "upright_only": False},
            {"id": "Watch", "w": 8, "h": 8, "d": 3, "weight": 100, "upright_only": False},
            {"id": "Digital Camera", "w": 12, "h": 8, "d": 7, "weight": 500, "upright_only": False},
            {"id": "Headphone1", "w": 20, "h": 15, "d": 10, "weight": 300, "upright_only": False},
            {"id": "Headphone2", "w": 20, "h": 15, "d": 10, "weight": 300, "upright_only": False},
        ]
    elif amountOfProducts == 6:
        products = [
            {"id": "Test1", "w": 2, "h": 2, "d": 3, "weight": 300, "upright_only": False},
            {"id": "Test2", "w": 2, "h": 2, "d": 3, "weight": 300, "upright_only": False},
            {"id": "Test3", "w": 2, "h": 2, "d": 3, "weight": 300, "upright_only": False},
            {"id": "Test4", "w": 2, "h": 2, "d": 3, "weight": 300, "upright_only": False},
            {"id": "Test5", "w": 2, "h": 2, "d": 3, "weight": 300, "upright_only": False},
            {"id": "Test6", "w": 2, "h": 2, "d": 3, "weight": 300, "upright_only": False},
        ]
    elif amountOfProducts == 7:
        products = [
            {"id": "Action Camera", "w": 8, "h": 6, "d": 10, "weight": 300, "upright_only": False},
            {"id": "Mini Drone", "w": 25, "h": 10, "d": 25, "weight": 900, "upright_only": False},         
        ]
    elif amountOfProducts == 8:
        products = [
            {"id": "Action Camera", "w": 8, "h": 6, "d": 10, "weight": 300, "upright_only": False},
            {"id": "Mini Drone", "w": 25, "h": 10, "d": 25, "weight": 900, "upright_only": False},
            {"id": "USB Flash Drive", "w": 2, "h": 2, "d": 8, "weight": 20, "upright_only": False},          
        ]
    elif amountOfProducts == 9:
        products = [
            {"id": "Action Camera", "w": 8, "h": 6, "d": 10, "weight": 300, "upright_only": False},
            {"id": "Mini Drone", "w": 25, "h": 10, "d": 25, "weight": 900, "upright_only": False},
            {"id": "USB Flash Drive", "w": 2, "h": 2, "d": 8, "weight": 20, "upright_only": False},
            {"id": "Fitness Tracker", "w":10, "h": 3, "d": 10, "weight": 100, "upright_only": False},          
        ]
    elif amountOfProducts == 10:
        products = [
            {"id": "Action Camera", "w": 8, "h": 6, "d": 10, "weight": 300, "upright_only": False},
            {"id": "Mini Drone", "w": 25, "h": 10, "d": 25, "weight": 900, "upright_only": False},
            {"id": "USB Flash Drive", "w": 2, "h": 2, "d": 8, "weight": 20, "upright_only": False},
            {"id": "Fitness Tracker", "w":10, "h": 3, "d": 10, "weight": 100, "upright_only": False},
            {"id": "Portable Speaker", "w":10, "h": 10, "d": 18, "weight": 800, "upright_only": False},            
        ]
    elif amountOfProducts == 11:
        products = [
            {"id": "Mini Drone", "w": 25, "h": 10, "d": 25, "weight": 900, "upright_only": False},
            {"id": "USB Flash Drive", "w": 2, "h": 2, "d": 8, "weight": 20, "upright_only": False},
            {"id": "Fitness Tracker", "w":10, "h": 3, "d": 10, "weight": 100, "upright_only": False},
            {"id": "Portable Speaker", "w":10, "h": 10, "d": 18, "weight": 800, "upright_only": False},            
        ]
    elif amountOfProducts == 12:
        products = [
            {"id": "USB Flash Drive", "w": 2, "h": 2, "d": 8, "weight": 20, "upright_only": False},
            {"id": "Fitness Tracker", "w":10, "h": 3, "d": 10, "weight": 100, "upright_only": False},
            {"id": "Portable Speaker", "w":10, "h": 10, "d": 18, "weight": 800, "upright_only": False},            
        ]
    elif amountOfProducts == 13:
        products = [
            {"id": "Portable Speaker1", "w":10, "h": 10, "d": 18, "weight": 800, "upright_only": False},            
            {"id": "Portable Speaker2", "w":10, "h": 10, "d": 18, "weight": 800, "upright_only": False},            
            {"id": "Portable Speaker3", "w":10, "h": 10, "d": 18, "weight": 800, "upright_only": False},            
        ]
    elif amountOfProducts == 14:
        products = [
            {"id": "Fitness Tracker1", "w":10, "h": 3, "d": 10, "weight": 100, "upright_only": False},
            {"id": "Fitness Tracker2", "w":10, "h": 3, "d": 10, "weight": 100, "upright_only": False},
            {"id": "Portable Speaker", "w":10, "h": 10, "d": 18, "weight": 800, "upright_only": False},            
        ]
    elif amountOfProducts == 15:
        products = [
            {"id": "Mini Drone1", "w": 25, "h": 10, "d": 25, "weight": 900, "upright_only": False},          
            {"id": "Mini Drone2", "w": 25, "h": 10, "d": 25, "weight": 900, "upright_only": False},          
        ]
    elif amountOfProducts == 16:
        products = [
            {"id": "Smartphone", "w": 15, "h": 7, "d": 1, "weight": 200, "upright_only": False},
            {"id": "Watch", "w": 8, "h": 8, "d": 3, "weight": 100, "upright_only": False},
            {"id": "Digital Camera1", "w": 12, "h": 8, "d": 7, "weight": 400, "upright_only": False},
            {"id": "Digital Camera2", "w": 12, "h": 8, "d": 7, "weight": 400, "upright_only": False},
            {"id": "Digital Camera3", "w": 12, "h": 8, "d": 7, "weight": 400, "upright_only": False},
            {"id": "Digital Camera4", "w": 12, "h": 8, "d": 7, "weight": 400, "upright_only": True},
            {"id": "Digital Camera5", "w": 12, "h": 8, "d": 7, "weight": 400, "upright_only": False},
            {"id": "Digital Camera6", "w": 12, "h": 8, "d": 7, "weight": 400, "upright_only": False},
            {"id": "Digital Camera7", "w": 12, "h": 8, "d": 7, "weight": 400, "upright_only": True},
            {"id": "Digital Camera8", "w": 12, "h": 8, "d": 7, "weight": 400, "upright_only": False},
            {"id": "Digital Camera9", "w": 12, "h": 8, "d": 7, "weight": 400, "upright_only": False},
        ]
    elif amountOfProducts == 17:
        products = [
            {"id": "Smartphone1", "w": 7, "h": 15, "d": 1, "weight": 200, "upright_only": True},
            {"id": "Smartphone2", "w": 7, "h": 15, "d": 1, "weight": 200, "upright_only": True},
            {"id": "Mini Drone", "w": 10, "h": 25, "d": 25, "weight": 900, "upright_only": True},
        ]  
    else:
        print("Keine Testdaten gefunden")
    
    return {
        "bins": containers,
        "products": products
    }  


def formatTime(start_time, end_time):
    # Calculate the time difference
    time_diff = end_time - start_time
    # Round the time to 3 decimal places
    time_in_seconds = time_diff.total_seconds()
    formatted_time = f"{time_in_seconds:.3f} seconds"

    return formatted_time

def run_test(iteration, iterationIntern, show_plot=False):
    # Erstelle Verzeichnis für Ergebnisse, falls es nicht existiert
    os.makedirs("./test_results", exist_ok=True)

    # Ergebnisliste für Dokumentation
    results_doc = []
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results_doc.append(f"Testlauf am {timestamp} \n{'='*50}\n")
    
    # documentiere container
    bins = _test_data(0)["bins"]
    results_doc.append(f"Container: {bins}")

    # Ergebnisse
    results_doc.append("Ergebnisse:")

    # Start Test
    index = 1
    while _test_data(index)["products"] != []:
        # Testdaten generieren
        print(index)
        test_data = _test_data(index)

        packer = EpbaBinPacker(container_definitions=test_data["bins"],)

        # Document products
        results_doc.append(f"Produkte: {test_data['products']} \n")
        amountOfProducts = len(test_data["products"])

        for idx in range(0, iteration):
            start_time = datetime.datetime.now()
            print(f"Testlauf:", idx, "von ", iteration, " - Anzahl Produkte:", amountOfProducts)

            # Führe den Algorithmus aus
            counts = start_test( 
                packer=packer,
                pack_iterations=iterationIntern,
                products=test_data["products"],
                show_plot=show_plot
            )
            end_time = datetime.datetime.now()

            # Dokumentiere diesen Test
            print(f"Amount of Prod: {amountOfProducts} - Time: {formatTime(start_time, end_time)} - {counts}")
            results_doc.append(f"Amount of Prod: {amountOfProducts} - Time: {formatTime(start_time, end_time)} - {counts}")

        # End
        print("---------------------------------------------")
        results_doc.append(f"{'='*130}")
        index += 1


    # Speichere die Dokumentation in eine Datei
    filepath = f"./test_results/ebpa_test_results_{timestamp.replace(':', '-').replace(' ', '_')}.txt"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(str(item) for item in results_doc))
    
    print("Die Testergebnisse wurden gespeichert.")


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    run_test(iteration=2, iterationIntern=2000, show_plot=False)
    end_time = datetime.datetime.now()
    print(f"Total Test 1 Time: {formatTime(start_time, end_time)}")

    # start_time = datetime.datetime.now()
    # run_test(iteration=10, iterationIntern=1000)
    # end_time = datetime.datetime.now()
    # print(f"Total Test 2 Time: {formatTime(start_time, end_time)}")

