import cv2
import numpy as np
from pathlib import Path
import json
import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

def create_test_set(base_image: np.ndarray, angles: List[float]) -> List[Tuple[np.ndarray, float]]:
    """
    Create a test set by rotating a base image at different angles.
    """
    test_set = []
    height, width = base_image.shape[:2]
    center = (width // 2, height // 2)
    
    for angle in angles:
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            base_image,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        test_set.append((rotated, angle))
    
    return test_set

def numpy_to_python_type(obj):
    """
    Convert numpy types to Python native types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def evaluate_rotation_detection(
    detect_rotation_angle,
    test_images: Dict[str, str],
    test_angles: List[float] = None
) -> Dict:
    """
    Evaluate rotation detection algorithm on multiple images.
    """
    if test_angles is None:
        test_angles = [-45, -30, -15, 0, 15, 30, 45]
    
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "images_tested": len(test_images),
        "angles_tested": len(test_angles),
        "per_image_results": {},
        "overall_metrics": {}
    }
    
    all_errors = []
    
    for image_name, image_path in test_images.items():
        base_image = cv2.imread(image_path)
        if base_image is None:
            print(f"Warning: Could not read image {image_path}")
            continue
            
        test_set = create_test_set(base_image, test_angles)
        
        image_results = {
            "angles_tested": [numpy_to_python_type(angle) for angle in test_angles],
            "detected_angles": [],
            "errors": [],
            "mean_error": 0,
            "max_error": 0
        }
        
        for rotated_image, true_angle in test_set:
            detected_angle = detect_rotation_angle(rotated_image)
            error = abs(true_angle - detected_angle)
            
            image_results["detected_angles"].append(numpy_to_python_type(detected_angle))
            image_results["errors"].append(numpy_to_python_type(error))
            all_errors.append(error)
        
        image_results["mean_error"] = numpy_to_python_type(np.mean(image_results["errors"]))
        image_results["max_error"] = numpy_to_python_type(np.max(image_results["errors"]))
        
        results["per_image_results"][image_name] = image_results
    
    results["overall_metrics"] = {
        "mean_error": numpy_to_python_type(np.mean(all_errors)),
        "max_error": numpy_to_python_type(np.max(all_errors)),
        "std_error": numpy_to_python_type(np.std(all_errors))
    }
    
    return results

def visualize_results(results: Dict, output_dir: str) -> None:
    """
    Create visualizations of evaluation results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    all_errors = []
    for image_results in results["per_image_results"].values():
        all_errors.extend(image_results["errors"])
    
    plt.hist(all_errors, bins=20, edgecolor='black')
    plt.title("Distribution of Angle Detection Errors")
    plt.xlabel("Error (degrees)")
    plt.ylabel("Frequency")
    plt.savefig(output_path / "error_distribution.png")
    plt.close()
    
    plt.figure(figsize=(12, 6))
    image_names = list(results["per_image_results"].keys())
    mean_errors = [results["per_image_results"][name]["mean_error"] 
                  for name in image_names]
    
    plt.bar(image_names, mean_errors)
    plt.title("Mean Error by Image")
    plt.xlabel("Image")
    plt.ylabel("Mean Error (degrees)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path / "mean_errors_by_image.png")
    plt.close()

def save_results(results: Dict, output_path: str) -> None:
    """
    Save evaluation results to a JSON file.
    """
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=numpy_to_python_type)