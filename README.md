# Edge Detection Pipeline Project

This project implements and compares different edge detection pipelines, including a general pipeline, image-specific pipelines, and the Canny edge detector.

## Prerequisites

Before running this project, ensure you have Python 3.7 or later installed on your system. You can download Python from [python.org](https://www.python.org/downloads/).

## Installation

1. Clone this repository or download the source code.

2. It's recommended to create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required libraries:
   ```
   pip install numpy opencv-python scikit-image scikit-learn
   ```

## Running the Code

1. Ensure your input images are in the correct directory. By default, the code looks for images in:
   ```
   ..\dataset
   ```
   You may need to modify this path in the main script to match your directory structure.

2. Run the main script:
   ```
   python run.py
   ```

3. The script will process all images in the specified directory, generate results, and create a LaTeX report.

4. Results will be saved in the `edge_detection_results` directory, including:
   - Processed images
   - A LaTeX report (`report.tex`) - This has already been used to generate the report attached
   - Individual metric files for each image

5. To generate a PDF from the LaTeX report, you'll need a LaTeX compiler installed on your system. If you have `pdflatex` available, the script will attempt to compile the report automatically. Otherwise, you can use an online LaTeX editor or a local LaTeX installation to compile the report manually.

## Troubleshooting

- If you encounter any "module not found" errors, ensure all required libraries are installed correctly.
- If images are not being processed, check that the image directory path in the main script is correct.
- For any other issues, please check the console output for error messages and ensure all prerequisites are met.

## Additional Notes

- The code uses a grid search for parameter optimization, which can be time-consuming for large datasets or parameter spaces.
- You can adjust the parameter grid in the main script to explore different optimization settings.
- The run time takes +- 15 minutes

If you have any questions or encounter any issues, please don't hesitate to ask for clarification or assistance.