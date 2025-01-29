# TableDataExtraction
StreamLit Table Detection Application

## Overview

The StreamLit Table Detection Application is designed to extract structured table data from images using an advanced pipeline that combines object detection, Optical Character Recognition (OCR), and AI-based data processing. The application utilizes a combination of Detectron2 for table detection, PaddleOCR for text extraction, and GPT-4 for structuring the extracted data.

## Components

1. Table Detection Module

Detection Model: Utilizes Detectron2 to detect tables in images and extract positional information.

Image Enhancement: Uses an image super-resolution model (Swin2SR) to enhance the quality of detected table images for improved OCR accuracy.

2. Data Extraction Module

OCR Processing: OCR extracts text from detected table images, supporting both English and Japanese.

AI-Powered Extraction: GPT processes extracted text and images to generate structured table data in JSON format.

3. Post-Processing Module

Data Cleaning & Validation: Enhances the accuracy and usability of extracted table data through alignment correction and format validation.

Export Options: Outputs final structured data in formats such as CSV or Excel.

Workflow

Image Upload: Users upload an image containing table data.

Table Detection: Detectron2 identifies tables and extracts positional data.

Image Enhancement: Swin2SR improves image quality for better text recognition.

OCR Processing: OCR extracts text from detected table images.

AI Data Structuring: GPT-4o (vision) processes text and image data, outputting structured JSON.

Post-Processing: The extracted data is refined and formatted.

Final Output: The cleaned table data is saved in the desired format.

## Improving Accuracy

To further enhance accuracy, future improvements may include:

Custom Model Training: Fine-tuning Detectron2 or exploring alternative detection models such as YOLO.

Optimized AI Processing: Refining prompt engineering for better GPT-4o (vision) performance.

Advanced OCR Solutions: Exploring additional OCR engines to improve recognition capabilities.

## License

This project is open-source and available for further development and contributions. Please refer to the LICENSE file for usage terms.

## Acknowledgments

Special thanks to the developers and contributors of Detectron2, Swin2SR, PaddleOCR, and OpenAI APIs for enabling this advanced table extraction workflow.
