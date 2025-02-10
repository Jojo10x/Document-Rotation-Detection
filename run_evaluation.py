from pathlib import Path
from rotation_detector import detect_rotation_angle
from evaluation import evaluate_rotation_detection, visualize_results, save_results

def run_evaluation():
    test_images_dir = Path("test_images")
    test_images = {}
    
    image_extensions = [".jpg", ".jpeg", ".png"]
    
    for image_path in test_images_dir.iterdir():
        if image_path.suffix.lower() in image_extensions:
            test_images[image_path.stem] = str(image_path)
    
    if not test_images:
        print("No images found in test_images directory!")
        return
    
    print(f"Found {len(test_images)} images for testing:")
    for name in test_images.keys():
        print(f"- {name}")
    
    test_angles = [-45, -30, -15, -5, 0, 5, 15, 30, 45]
    
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    print("\nRunning evaluation...")
    results = evaluate_rotation_detection(
        detect_rotation_angle,
        test_images,
        test_angles
    )
    
    save_results(results, output_dir / "results.json")
    
    visualize_results(results, output_dir)
    
    print("\nEvaluation Summary:")
    print(f"Images tested: {results['images_tested']}")
    print(f"Angles tested: {results['angles_tested']}")
    print(f"Overall mean error: {results['overall_metrics']['mean_error']:.2f} degrees")
    print(f"Overall max error: {results['overall_metrics']['max_error']:.2f} degrees")
    print(f"Error standard deviation: {results['overall_metrics']['std_error']:.2f} degrees")
    
    print("\nResults have been saved to:")
    print(f"- JSON results: {output_dir}/results.json")
    print(f"- Plots: {output_dir}/error_distribution.png and mean_errors_by_image.png")

if __name__ == "__main__":
    run_evaluation()