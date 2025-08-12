# PDF Comparison & Data Extraction System

## Overview
This project provides a set of Python modules for:
- Extracting and normalizing tables from PDF documents (including borderless tables)
- Comparing images, tables, and text across different PDF versions
- Managing PDF data and extracted elements in a PostgreSQL database
- Identifying and visualizing changes between document revisions

It is designed for workflows where raw PDF files are ingested into a database, and their structure, images, and textual content need to be compared during testing or quality assurance flows.

---

## Features
- **Image Comparison**  
  Extracts figures from PDFs, matches them by name and position, and identifies differences at the pixel level with support for:
  - Pixel change detection
  - Brightness and color shift analysis
  - Structural difference detection

- **Table Extraction & Comparison**  
  - Uses `camelot` and `pdfplumber` to extract structured table data
  - Normalizes headers, merges split tables, and removes noise
  - Compares tables cell-by-cell across versions

- **Text Comparison**  
  - Matches text lines by TOC context, page, and position
  - Highlights content differences between versions

- **Database Integration**  
  - Stores extracted images and tables in PostgreSQL schemas
  - Fetches stored versions for comparison
  - Manages schema organization for different PDF versions

---

## Directory Structure
- ├── compare_image.py # Image extraction and comparison
- ├── compare_table.py # Table extraction and comparison
- ├── compare_texts.py # Text and TOC comparison
- ├── database_management.py # PostgreSQL operations
- ├── extract_normalize_tables.py # Table cleaning & normalization
- ├── pdf_management.py # PDF parsing utilities
- ├── processing_lines.py # PDF line detection for table structure


## Installation

### 1. Clone the Repository
- git clone https://github.com/yourusername/pdf-comparison-tool.git
- cd pdf-comparison-tool
### 2. Create & Activate Virtual Environment
- python -m venv venv
- source venv/bin/activate    # On macOS/Linux
- venv\Scripts\activate       # On Windows 

### 3. Install Dependencies
- pip install -r requirements.txt

### 4. Set Up Environment Variables
Create a .env file with your database configuration:

- DB_HOST=localhost, 
- DB_PORT=5432, 
- DB_NAME_ERS=ers_database, 
- DB_NAME_LWS=lws_database, 
- DB_NAME_PTC=ptc_database, 
- DB_NAME_PDF=pdf_database, 
- DB_USER=your_username, 
- DB_PASSWORD=your_password

### Usage
 Here’s a quick example of comparing two PDFs for image differences:

 from compare_image import convert_img_name

 pdf_list = ["v1_document.pdf", "v2_document.pdf"]
 conversion_map, images = convert_img_name(pdf_list)
 print(conversion_map)
 Extracting and comparing tables:

 from compare_table import extract_tables, compare_tables

 tables_v1 = extract_tables("v1_document.pdf")
 tables_v2 = extract_tables("v2_document.pdf")
 diff_table = compare_tables([tables_v1, tables_v2])
 print(diff_table ) 

### Dependencies
See requirements.txt for a full list of dependencies.
