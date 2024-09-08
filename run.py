import cv2
import os
import numpy as np
from typing import Dict, Any, Tuple, List
from itertools import product
import time
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error

class EdgeDetectionStep:
    def __init__(self, params: Dict[str, Any]):
        self.params = params

    def process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This method should be implemented by subclasses")

class NoiseReduction(EdgeDetectionStep):
    def process(self, image: np.ndarray) -> np.ndarray:
        if self.params['method'] == 'gaussian':
            return cv2.GaussianBlur(image, 
                                    (self.params['kernel_size'],)*2, 
                                    self.params['sigma'])
        elif self.params['method'] == 'median':
            return cv2.medianBlur(image, self.params['kernel_size'])
        else:
            raise ValueError(f"Unknown noise reduction method: {self.params['method']}")

class GradientCalculation(EdgeDetectionStep):
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.params['method'] == 'sobel':
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            direction = np.arctan2(grad_y, grad_x)
            return magnitude, direction
        else:
            raise ValueError(f"Unknown gradient calculation method: {self.params['method']}")

class EdgeThinning(EdgeDetectionStep):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.params.setdefault('min_threshold', 10)
        self.params.setdefault('max_threshold', 200)

    def process(self, inputs: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        magnitude, direction = inputs
        h, w = magnitude.shape
        result = np.zeros((h, w), dtype=np.float32)
        direction = direction * 180. / np.pi
        direction[direction < 0] += 180

        min_threshold = self.params['min_threshold']
        max_threshold = self.params['max_threshold']

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                try:
                    q = 255
                    r = 255

                    # angle 0
                    if (0 <= direction[i, j] < 22.5) or (157.5 <= direction[i, j] <= 180):
                        q = magnitude[i, j+1]
                        r = magnitude[i, j-1]
                    # angle 45
                    elif (22.5 <= direction[i, j] < 67.5):
                        q = magnitude[i+1, j-1]
                        r = magnitude[i-1, j+1]
                    # angle 90
                    elif (67.5 <= direction[i, j] < 112.5):
                        q = magnitude[i+1, j]
                        r = magnitude[i-1, j]
                    # angle 135
                    elif (112.5 <= direction[i, j] < 157.5):
                        q = magnitude[i-1, j-1]
                        r = magnitude[i+1, j+1]

                    if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                        if magnitude[i, j] >= max_threshold:
                            result[i, j] = magnitude[i, j]
                        elif magnitude[i, j] >= min_threshold:
                            result[i, j] = min_threshold

                except IndexError:
                    pass

        return result

class Thresholding(EdgeDetectionStep):
    def process(self, image: np.ndarray) -> np.ndarray:
        if self.params['method'] == 'hysteresis':
            low, high = self.params['low'], self.params['high']
            strong_edges = (image > high)
            weak_edges = (image >= low) & (image <= high)
            result = np.zeros_like(image, dtype=np.uint8)
            result[strong_edges] = 255
            return self.hysteresis(result, weak_edges)
        else:
            raise ValueError(f"Unknown thresholding method: {self.params['method']}")

    def hysteresis(self, strong_edges: np.ndarray, weak_edges: np.ndarray) -> np.ndarray:
        h, w = strong_edges.shape
        for i in range(1, h-1):
            for j in range(1, w-1):
                if weak_edges[i, j]:
                    if np.any(strong_edges[i-1:i+2, j-1:j+2]):
                        strong_edges[i, j] = 255
        return strong_edges

class EdgeLinking(EdgeDetectionStep):
    def process(self, image: np.ndarray) -> np.ndarray:
        if self.params['method'] == 'simple':
            return self.simple_linking(image)
        else:
            raise ValueError(f"Unknown edge linking method: {self.params['method']}")

    def simple_linking(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape
        result = np.zeros((h, w), dtype=np.uint8)
        for i in range(1, h-1):
            for j in range(1, w-1):
                if image[i, j] == 255:
                    if np.sum(image[i-1:i+2, j-1:j+2]) >= 2 * 255:
                        result[i, j] = 255
        return result

class EdgeDetectionPipeline:
    def __init__(self, steps: Dict[str, EdgeDetectionStep]):
        self.steps = steps

    def process(self, image: np.ndarray) -> np.ndarray:
        result = image
        grad_mag, grad_dir = None, None
        for step_name, step in self.steps.items():
            if step_name == 'gradient_calculation':
                grad_mag, grad_dir = step.process(result)
            elif step_name == 'edge_thinning':
                result = step.process((grad_mag, grad_dir))
            else:
                result = step.process(result)
        return result

    def get_params(self) -> Dict[str, Dict[str, Any]]:
        return {name: step.params for name, step in self.steps.items()}

    def set_params(self, params: Dict[str, Dict[str, Any]]):
        for name, step_params in params.items():
            if name in self.steps:
                self.steps[name].params.update(step_params)

def compute_all_metrics(original: np.ndarray, processed: np.ndarray) -> Dict[str, float]:
    feature_sim_orb = compute_feature_similarity(original, processed, method='ORB')
    feature_sim_sift = compute_feature_similarity(original, processed, method='SIFT')
    mse = compute_mse(original, processed)
    psnr = compute_psnr(original, processed)
    ssim_score = compute_ssim(original, processed)
    
    return {
        'ORB_similarity': feature_sim_orb,
        'SIFT_similarity': feature_sim_sift,
        'MSE': mse,
        'PSNR': psnr,
        'SSIM': ssim_score
    }

def evaluate_pipeline(original: np.ndarray, processed: np.ndarray) -> float:
    metrics = compute_all_metrics(original, processed)
    
    # Normalize MSE (lower is better, so we invert it)
    max_mse = 255 ** 2  # Maximum possible MSE for 8-bit images
    norm_mse = 1 - (metrics['MSE'] / max_mse)
    
    # Normalize PSNR
    max_psnr = 100  # Assuming this as a reasonable maximum PSNR
    norm_psnr = min(metrics['PSNR'] / max_psnr, 1)
    
    # Combine metrics (you may adjust weights as needed)
    combined_score = (
        0.2 * metrics['ORB_similarity'] +
        0.2 * metrics['SIFT_similarity'] +
        0.2 * norm_mse +
        0.2 * norm_psnr +
        0.2 * metrics['SSIM']
    )
    
    return combined_score

def create_default_pipeline() -> EdgeDetectionPipeline:
    return EdgeDetectionPipeline({
        'noise_reduction': NoiseReduction({'method': 'gaussian', 'kernel_size': 5, 'sigma': 1}),
        'gradient_calculation': GradientCalculation({'method': 'sobel'}),
        'edge_thinning': EdgeThinning({'min_threshold': 10, 'max_threshold': 200}),
        'thresholding': Thresholding({'method': 'hysteresis', 'low': 50, 'high': 100}),
        'edge_linking': EdgeLinking({'method': 'simple'})
    })

def compute_feature_similarity(img1: np.ndarray, img2: np.ndarray, method: str = 'ORB') -> float:
    if method == 'ORB':
        feature_extractor = cv2.ORB_create()
        norm = cv2.NORM_HAMMING
    elif method == 'SIFT':
        feature_extractor = cv2.SIFT_create()
        norm = cv2.NORM_L2
    else:
        raise ValueError(f"Unknown feature extraction method: {method}")
    
    # Detect and compute keypoints and descriptors
    kp1, des1 = feature_extractor.detectAndCompute(img1, None)
    kp2, des2 = feature_extractor.detectAndCompute(img2, None)
    
    # Handle cases where no keypoints are detected
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return 0.0
    
    # Match descriptors
    bf = cv2.BFMatcher(norm, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Calculate similarity score based on the number of good matches
    similarity = len(matches) / max(len(kp1), len(kp2))
    return similarity

def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    return mean_squared_error(img1, img2)

def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = compute_mse(img1, img2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    try:
        return ssim(img1, img2, data_range=img2.max() - img2.min())
    except RuntimeWarning:
        # If SSIM calculation fails, return a default low similarity score
        return 0.0

def evaluate_pipeline(original: np.ndarray, processed: np.ndarray) -> float:
    # Compute all metrics
    feature_sim = compute_feature_similarity(original, processed, method='ORB')
    mse = compute_mse(original, processed)
    psnr = compute_psnr(original, processed)
    ssim_score = compute_ssim(original, processed)
    
    # Normalize MSE (lower is better, so we invert it)
    max_mse = 255 ** 2  # Maximum possible MSE for 8-bit images
    norm_mse = 1 - (mse / max_mse)
    
    # Normalize PSNR
    max_psnr = 100  # Assuming this as a reasonable maximum PSNR
    norm_psnr = min(psnr / max_psnr, 1)
    
    # Combine metrics (you may adjust weights as needed)
    combined_score = (
        0.25 * feature_sim +
        0.25 * norm_mse +
        0.25 * norm_psnr +
        0.25 * ssim_score
    )
    
    return combined_score

def grid_search_optimizer(image: np.ndarray, pipeline: EdgeDetectionPipeline, param_grid: Dict[str, Dict[str, List[Any]]], eval_func) -> Tuple[EdgeDetectionPipeline, float]:
    best_score = float('-inf')
    best_params = None
    
    # Generate all possible combinations of parameters
    param_names = []
    param_values = []
    for step_name, step_params in param_grid.items():
        for param_name, param_value_list in step_params.items():
            param_names.append((step_name, param_name))
            param_values.append(param_value_list)
    
    total_combinations = np.prod([len(values) for values in param_values])
    print(f"Total parameter combinations to try: {total_combinations}")
    
    for i, combination in enumerate(product(*param_values)):
        # Update pipeline parameters
        new_params = pipeline.get_params()
        for (step_name, param_name), value in zip(param_names, combination):
            if step_name not in new_params:
                new_params[step_name] = {}
            new_params[step_name][param_name] = value
        pipeline.set_params(new_params)
        
        # Process image and evaluate
        result = pipeline.process(image)
        score = eval_func(image, result)
        
        # Update best parameters if necessary
        if score > best_score:
            best_score = score
            best_params = new_params
        
        # Print progress
        if (i + 1) % 10 == 0 or (i + 1) == total_combinations:
            print(f"Processed {i+1}/{total_combinations} combinations.")
    
    # Set the best parameters in the pipeline
    pipeline.set_params(best_params)
    return pipeline, best_score

def process_image_set(image_paths: List[str], general_pipeline: EdgeDetectionPipeline, param_grid: Dict[str, Dict[str, List[Any]]]) -> Dict[str, Dict[str, Any]]:
    results = {}
    
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        print(f"\nProcessing image: {img_name}")
        
        # Load image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Process with general pipeline
        general_result = general_pipeline.process(image)
        general_metrics = compute_all_metrics(image, general_result)
        general_score = evaluate_pipeline(image, general_result)
        
        # Optimize image-specific pipeline
        specific_pipeline = create_default_pipeline()
        optimized_specific_pipeline, specific_score = grid_search_optimizer(image, specific_pipeline, param_grid, evaluate_pipeline)
        specific_result = optimized_specific_pipeline.process(image)
        specific_metrics = compute_all_metrics(image, specific_result)
        
        # Canny Edge Detector
        canny_result = cv2.Canny(image, 100, 200)
        canny_metrics = compute_all_metrics(image, canny_result)
        canny_score = evaluate_pipeline(image, canny_result)
        
        # Store results
        results[img_name] = {
            'original': image,
            'general': {
                'result': general_result,
                'score': general_score,
                'metrics': general_metrics
            },
            'specific': {
                'result': specific_result,
                'score': specific_score,
                'metrics': specific_metrics,
                'params': optimized_specific_pipeline.get_params()
            },
            'canny': {
                'result': canny_result,
                'score': canny_score,
                'metrics': canny_metrics
            }
        }
        
        print(f"General pipeline score: {general_score:.4f}")
        print(f"Image-specific pipeline score: {specific_score:.4f}")
        print(f"Canny Edge Detector score: {canny_score:.4f}")
    
    return results

def save_results(results: Dict[str, Dict[str, Any]], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    for img_name, data in results.items():
        base_name = os.path.splitext(img_name)[0]
        
        # Save images
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_original.png"), data['original'])
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_general.png"), data['general']['result'])
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_specific.png"), data['specific']['result'])
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_canny.png"), data['canny']['result'])
        
        # Save metrics
        with open(os.path.join(output_dir, f"{base_name}_metrics.txt"), 'w') as f:
            f.write(f"General pipeline score: {data['general']['score']:.4f}\n")
            f.write(f"Image-specific pipeline score: {data['specific']['score']:.4f}\n")
            f.write(f"Canny Edge Detector score: {data['canny']['score']:.4f}\n")
            f.write("\nImage-specific pipeline parameters:\n")
            for step_name, params in data['specific']['params'].items():
                f.write(f"  {step_name}: {params}\n")

def generate_report(results: Dict[str, Dict[str, Any]], output_file: str):
    with open(output_file, 'w') as f:
        f.write("# Edge Detection Pipeline Comparison Report\n\n")
        
        # Overall summary
        f.write("## Overall Summary\n\n")
        f.write("| Image | General Pipeline | Image-Specific Pipeline | Canny Edge Detector |\n")
        f.write("|-------|-------------------|-------------------------|----------------------|\n")
        for img_name, data in results.items():
            f.write(f"| {img_name} | {data['general']['score']:.4f} | {data['specific']['score']:.4f} | {data['canny']['score']:.4f} |\n")
        
        f.write("\n## Detailed Results\n\n")
        for img_name, data in results.items():
            f.write(f"### {img_name}\n\n")
            
            for pipeline in ['general', 'specific', 'canny']:
                f.write(f"#### {pipeline.capitalize()} Pipeline\n\n")
                f.write(f"Overall Score: {data[pipeline]['score']:.4f}\n\n")
                f.write("Individual Metrics:\n")
                if 'metrics' in data[pipeline]:
                    for metric, value in data[pipeline]['metrics'].items():
                        f.write(f"- {metric}: {value:.4f}\n")
                else:
                    f.write("Metrics not available for this pipeline.\n")
                f.write("\n")
            
            f.write("Image-Specific Pipeline Parameters:\n")
            for step_name, params in data['specific']['params'].items():
                f.write(f"- {step_name}:\n")
                for param_name, param_value in params.items():
                    f.write(f"  - {param_name}: {param_value}\n")
            f.write("\n")

def generate_latex_report(results: Dict[str, Dict[str, Any]], output_file: str, param_grid: Dict[str, Dict[str, List[Any]]]):
    with open(output_file, 'w') as f:
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage{graphicx}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\usepackage{float}\n")
        f.write("\\usepackage{longtable}\n")
        f.write("\\usepackage{hyperref}\n")
        f.write("\\title{Edge Detection Pipeline Comparison Report}\n")
        f.write("\\author{Your Name}\n")
        f.write("\\date{\\today}\n\n")
        f.write("\\begin{document}\n\n")
        f.write("\\maketitle\n\n")
        
        # Table of Contents
        f.write("\\tableofcontents\n\n")
        
        # Introduction
        f.write("\\section{Introduction}\n")
        f.write("This report presents a comparison of different edge detection pipelines, including a general pipeline, image-specific pipelines, and the Canny edge detector. "
                "The performance of these pipelines is evaluated using various metrics on a set of test images.\n\n")
        
        # Methodology
        f.write("\\section{Methodology}\n")
        f.write("\\subsection{Pipeline Components}\n")
        f.write("The edge detection pipeline consists of the following components:\n")
        f.write("\\begin{itemize}\n")
        f.write("\\item Noise Reduction\n")
        f.write("\\item Gradient Calculation\n")
        f.write("\\item Edge Thinning\n")
        f.write("\\item Thresholding\n")
        f.write("\\item Edge Linking\n")
        f.write("\\end{itemize}\n\n")
        
        f.write("\\subsection{Optimization Process}\n")
        f.write("The pipeline parameters were optimized using a grid search algorithm. The following parameter grid was used:\n")
        f.write("\\begin{verbatim}\n")
        f.write(str(param_grid))
        f.write("\\end{verbatim}\n\n")
        
        f.write("\\subsection{Evaluation Metrics}\n")
        f.write("The following metrics were used to evaluate the performance of the edge detection pipelines:\n")
        f.write("\\begin{itemize}\n")
        f.write("\\item ORB Feature Similarity\n")
        f.write("\\item SIFT Feature Similarity\n")
        f.write("\\item Mean Squared Error (MSE)\n")
        f.write("\\item Peak Signal-to-Noise Ratio (PSNR)\n")
        f.write("\\item Structural Similarity Index (SSIM)\n")
        f.write("\\end{itemize}\n\n")
        
        # Results
        f.write("\\section{Results}\n")
        f.write("\\subsection{Overall Performance}\n")
        f.write("\\begin{longtable}{lrrr}\n")
        f.write("\\toprule\n")
        f.write("Image & General Pipeline & Image-Specific Pipeline & Canny Edge Detector \\\\\n")
        f.write("\\midrule\n")
        for img_name, data in results.items():
            f.write(f"{img_name} & {data['general']['score']:.4f} & {data['specific']['score']:.4f} & {data['canny']['score']:.4f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{longtable}\n\n")
        
        f.write("\\subsection{Detailed Results}\n")
        for img_name, data in results.items():
            f.write(f"\\subsubsection{{{img_name}}}\n")
            
            # Add images
            f.write("\\begin{figure}[H]\n")
            f.write("\\centering\n")
            f.write(f"\\includegraphics[width=0.24\\textwidth]{{{img_name}_original.png}}\n")
            f.write(f"\\includegraphics[width=0.24\\textwidth]{{{img_name}_general.png}}\n")
            f.write(f"\\includegraphics[width=0.24\\textwidth]{{{img_name}_specific.png}}\n")
            f.write(f"\\includegraphics[width=0.24\\textwidth]{{{img_name}_canny.png}}\n")
            f.write("\\caption{Edge detection results for " + img_name + ". From left to right: Original, General Pipeline, Image-Specific Pipeline, Canny Edge Detector}\n")
            f.write("\\end{figure}\n\n")
            
            # Add metric tables
            for pipeline in ['general', 'specific', 'canny']:
                f.write(f"\\paragraph{{{pipeline.capitalize()} Pipeline}}\n")
                f.write("\\begin{tabular}{lr}\n")
                f.write("\\toprule\n")
                f.write("Metric & Value \\\\\n")
                f.write("\\midrule\n")
                for metric, value in data[pipeline]['metrics'].items():
                    f.write(f"{metric} & {value:.4f} \\\\\n")
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n\n")
            
            # Add image-specific pipeline parameters
            f.write("\\paragraph{Image-Specific Pipeline Parameters}\n")
            f.write("\\begin{verbatim}\n")
            f.write(str(data['specific']['params']))
            f.write("\\end{verbatim}\n\n")
        
        # Critical Analysis
        f.write("\\section{Critical Analysis}\n")
        f.write("Based on the results obtained, we can make the following observations:\n")
        f.write("\\begin{itemize}\n")
        f.write("\\item The image-specific pipeline generally outperforms the general pipeline, demonstrating the benefits of parameter tuning for individual images.\n")
        f.write("\\item The Canny edge detector, despite its simplicity, performs competitively with our custom pipelines. This suggests that there might be room for improvement in our pipeline design or optimization process.\n")
        f.write("\\item [Add more insights based on your specific results]\n")
        f.write("\\end{itemize}\n\n")
        
        f.write("\\section{Conclusion}\n")
        f.write("This study compared different edge detection approaches, including a general pipeline, image-specific pipelines, and the Canny edge detector. "
                "The results highlight the importance of parameter tuning in edge detection tasks and provide insights into the strengths and weaknesses of each approach. "
                "Future work could focus on improving the optimization process, exploring more advanced edge detection techniques, or investigating the performance of these pipelines on a broader range of image types.\n")
        
        f.write("\\end{document}\n")

    print(f"LaTeX report generated: {output_file}")

if __name__ == "__main__":
    # Define image directory
    image_dir = r'..\dataset'
    
    # Get all image paths
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_paths:
        print("No image files found in the specified directory.")
        exit()

    print(f"Found {len(image_paths)} images to process.")

    # Create and optimize general pipeline on the first image
    general_pipeline = create_default_pipeline()
    first_image = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)

    if first_image is None:
        print(f"Failed to load the first image: {image_paths[0]}")
        exit()

    param_grid = {
        'noise_reduction': {
            'method': ['gaussian', 'median'],
            'kernel_size': [3, 5, 7],
            'sigma': [0.5, 1, 1.5]
        },
        'edge_thinning': {
            'min_threshold': [5, 10, 15],
            'max_threshold': [150, 200, 250]
        },
        'thresholding': {
            'low': [30, 50, 70],
            'high': [80, 100, 120]
        }
    }

    print("Optimizing general pipeline...")
    optimized_general_pipeline, _ = grid_search_optimizer(first_image, general_pipeline, param_grid, evaluate_pipeline)

    # Process all images
    print("Processing all images...")
    results = process_image_set(image_paths, optimized_general_pipeline, param_grid)

    # Create output directory if it doesn't exist
    output_dir = 'edge_detection_results'
    os.makedirs(output_dir, exist_ok=True)

    # Save results (images and metrics)
    print(f"Saving results to {output_dir}...")
    save_results(results, output_dir)

    # Generate LaTeX report
    latex_report_path = os.path.join(output_dir, 'report.tex')
    print(f"Generating LaTeX report: {latex_report_path}")
    generate_latex_report(results, latex_report_path, param_grid)

    print(f"\nProcessing complete. Results saved in {output_dir}")
    print(f"LaTeX report generated: {latex_report_path}")
    print("Compile the LaTeX report to generate a PDF.")

    # Optional: Automatically compile the LaTeX report if pdflatex is available
    try:
        import subprocess
        print("Attempting to compile the LaTeX report...")
        subprocess.run(['pdflatex', '-output-directory', output_dir, latex_report_path], check=True)
        print(f"PDF report generated: {os.path.splitext(latex_report_path)[0]}.pdf")
    except Exception as e:
        print(f"Could not automatically compile the LaTeX report: {e}")
        print("Please compile the LaTeX report manually to generate a PDF.")