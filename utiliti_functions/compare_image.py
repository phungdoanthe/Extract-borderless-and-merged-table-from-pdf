from dataclasses import dataclass
from enum import Enum
import base64
import io
import fitz 
import pdfplumber
import pandas as pd
import numpy as np
import cv2
from typing import *
from PIL import Image, ImageDraw, ImageFont
from difflib import SequenceMatcher
from .pdf_management import *
from .database_management import *

def get_all_figure_name(pdf):
    page_len = get_page_info(pdf)[1]
    with pdfplumber.open(pdf) as pdf:
        extracted_text_lines = pdf.pages[0].extract_text_lines(
                                                char_margin =  2,   
                                                word_margin = 0.5)
        lines = pd.json_normalize(extracted_text_lines)
        lines['page'] = 1    
        for page_no, page in enumerate(pdf.pages):
                if page_no < 2:
                        continue
                extracted_text_lines = page.extract_text_lines() 
                line = pd.json_normalize(extracted_text_lines)
                line['page'] = page_no+1
                if isinstance(extracted_text_lines, list):
                        lines = pd.concat([lines, line], ignore_index=True)
                else:
                        raise ValueError("Expected a list of dictionaries from page.extract_text_lines()")  
        figure_name = lines.loc[(lines['text'].str.contains(r'^Figure ', regex=True))][['text', 'page', 'top']] #|\
                                # lines['text'].str.contains(r'^TTaabbllee', regex=True)) |\
                                # lines['text'].str.contains(r'^Relay Lens Set-Up Requirements', regex=True)]\
                                        
        figure_name['text'] = figure_name['text'].apply(lambda x: x.split(" Rev ")[0] if " Rev " in x else x)
        figure_name['top'] = figure_name['top'].astype(int).apply(lambda x: page_len-int(x))
        figure_name.rename(columns={'text': 'figure'}, inplace=True)
        figure_name['figure'] = figure_name['figure'].apply(lambda x: refine_table_name(x))
        figure_name = figure_name.reset_index(drop=True)
        return figure_name

def extract_images(pdf_path):
    ver = pdf_path.split("\\")[-1].split(".")[0]
    doc = fitz.open(pdf_path)
    figure_names = get_all_figure_name(pdf_path)

    
    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)
        figure_names_this_page = figure_names[figure_names['page']==page_num+1]
        
        if not images:
            continue
        
        image_infor = []
        for index, img in enumerate(images):
            xref = img[0]
            rects = page.get_image_rects(xref)
            
            if not rects:  # Skip if no visible rectangles
                continue
            rect = rects[0]
            image_infor.append({
                'xref': xref,
                'rect': rect,
                'y0': 792-rect[1],
            })
        # Sort images by vertical position (top to bottom)
        image_infor.sort(key=lambda x: x["y0"], reverse=True)

        image_files = []
        for i, infor in enumerate(image_infor):
            xref = infor['xref']


            if len(figure_names_this_page) == len(images):
                figure_name = figure_names_this_page.iloc[i]['figure']
            else:
                current_page_matches = figure_names[
                    (figure_names['page'] == page_num+1) & 
                    (
                        (abs(figure_names['top'] - infor['y0']) < 50) | 
                        (abs((792 - infor['rect'][3]) - figure_names['top']) < 55)
                    )
                ]

                next_page_matches = figure_names[
                    (figure_names['page'] == page_num+2) & 
                    (figure_names['top'] > 700)
                ]

                # Combine the matches
                figure_names_match = pd.concat([current_page_matches, next_page_matches])

                if not figure_names_match.empty:
                    figure_name = figure_names_match.iloc[-1]['figure']
                else:
                    figure_name = f"page-{page_num + 1}-image-{i+1}"

            figure_name = figure_name.replace(":", "")

            figure_name = figure_name + f"_{i+1}"

            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            img_ext = base_image["ext"]

            # save the image to database
            save_images_to_db(figure_name, img_bytes, img_ext, ver)

def convert_img_name(pdf_list):
    if not pdf_list or len(pdf_list) <2:
        return {}, []

    schema_names = []
    image_names_list = []

    existing_schemas = get_schema_names(host=host, port=port, database=dbname_PTC, username=username, password=password)

    for pdf in pdf_list:
        schema_name = pdf.split("\\")[-1].split(".")[0]

        if schema_name not in existing_schemas:
            extract_images(pdf)
        schema_names.append(schema_name)

        image_names = get_images_name(schema_name)
        image_names_list.append(image_names)

    pdf1 = schema_names[0]
    pdf2 = schema_names[-1]

    img_name_df1 = pd.DataFrame(image_names_list[0])
    img_name_df1.columns = [pdf1]

    img_name_df2 = pd.DataFrame(image_names_list[-1])
    img_name_df2.columns = [pdf2]

    cross_img_name_df = img_name_df1.merge(img_name_df2, how='cross')
    same_img_name_list = cross_img_name_df[cross_img_name_df.nunique(axis=1) == 1].iloc[:,0].tolist()
    diff_img_name_df = cross_img_name_df[cross_img_name_df.nunique(axis=1) != 1]

    diff_img_name_df = diff_img_name_df[~diff_img_name_df[pdf1].isin(same_img_name_list)]

    diff_img_name_df['similarity'] = diff_img_name_df.apply(lambda row: SequenceMatcher(None, str(row[pdf1]), str(row[pdf2])).ratio(), axis=1)

    top_left = diff_img_name_df.groupby(schema_names[0]).agg({'similarity': 'max'})
    top_right = diff_img_name_df.groupby(schema_names[-1]).agg({'similarity': 'max'})

    diff_img_name_df = diff_img_name_df.merge(top_left)
    diff_img_name_df = diff_img_name_df.merge(top_right)

    if len(img_name_df1) >= len(img_name_df2):
        convert_dict = dict(zip(diff_img_name_df[pdf1], diff_img_name_df[pdf2]))
        image_names = image_names_list[0]
    else:
        convert_dict = dict(zip(diff_img_name_df[pdf2], diff_img_name_df[pdf1]))
        image_names = image_names_list[-1]

    return convert_dict, image_names

class DifferenceType(Enum):
    """Types of differences that can be detected"""
    PIXEL_CHANGE = "pixel_change"
    BRIGHTNESS = "brightness"
    COLOR_SHIFT = "color_shift"
    STRUCTURAL = "structural"
    MISSING_REGION = "missing_region"

@dataclass
class ImageMetadata:
    """Core metadata for each image"""
    width: int
    height: int
    channels: int  # RGB=3, RGBA=4, Grayscale=1
    format: str
    file_path: str
    file_size: int
    
class ImageData:
    """Main container for image information"""
    def __init__(self, image_path: str):
        self.metadata = self._extract_metadata(image_path)
        self.pixel_array = None  # Will store numpy array
        self.histogram = None    # Color distribution
        self.hash_value = None   # For quick similarity checks
        
    def load_pixels(self) -> np.ndarray:
        """Load image as numpy array for pixel-level comparison"""
        if self.pixel_array is None:
            img = Image.open(self.metadata.file_path)
            # Convert to RGB if needed for consistent comparison
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            elif img.mode == 'L':
                img = img.convert('RGB')  # Convert grayscale to RGB for consistency
            self.pixel_array = np.array(img)
        return self.pixel_array
    
    def _extract_metadata(self, path: str) -> ImageMetadata:
        """Extract basic image information"""
        import os
        img = Image.open(path)
        
        return ImageMetadata(
            width=img.width,
            height=img.height,
            channels=len(img.getbands()),
            format=img.format or 'UNKNOWN',
            file_path=path,
            file_size=os.path.getsize(path)
        )

@dataclass
class DifferenceRegion:
    """Represents a region where images differ"""
    x: int
    y: int
    width: int
    height: int
    difference_type: DifferenceType
    confidence: float  # 0.0 to 1.0
    pixel_count: int
    avg_intensity_diff: float
    
class ComparisonResult:
    """Complete result of image comparison"""
    def __init__(self):
        self.overall_similarity: float = 0.0  # 0.0 = completely different, 1.0 = identical
        self.difference_regions: List[DifferenceRegion] = []
        self.difference_mask: np.ndarray = None  # Binary mask showing differences
        self.heatmap: np.ndarray = None  # Intensity map of differences
        self.stats: Dict[str, Any] = {}
        
    def add_difference_region(self, region: DifferenceRegion):
        """Add a detected difference region"""
        self.difference_regions.append(region)
    
    def get_total_changed_pixels(self) -> int:
        """Count total pixels that changed"""
        return sum(region.pixel_count for region in self.difference_regions)

class ImageComparator:
    """Main class for comparing two images"""
    
    def __init__(self, tolerance: float = 0.1):
        self.tolerance = tolerance  # Threshold for considering pixels different
        self.comparison_methods = []
        
    def compare(self, image1: ImageData, image2: ImageData) -> ComparisonResult:
        """Compare two images and return detailed results"""
        result = ComparisonResult()
        
        # Load pixel data first
        pixels1 = image1.load_pixels()
        pixels2 = image2.load_pixels()
        
        # Ensure both images have same shape
        if pixels1.shape != pixels2.shape:
            # Resize smaller image to match larger one
            from PIL import Image as PILImage
            h1, w1 = pixels1.shape[:2]
            h2, w2 = pixels2.shape[:2]
            
            if h1 * w1 < h2 * w2:  # pixels1 is smaller
                img1_resized = PILImage.fromarray(pixels1).resize((w2, h2), PILImage.LANCZOS)
                pixels1 = np.array(img1_resized)
            else:  # pixels2 is smaller
                img2_resized = PILImage.fromarray(pixels2).resize((w1, h1), PILImage.LANCZOS)
                pixels2 = np.array(img2_resized)
        
        # Ensure both have same number of channels
        if len(pixels1.shape) != len(pixels2.shape):
            raise ValueError("Images must have same format (both color or both grayscale)")
        
        # Perform pixel-wise comparison
        result.difference_mask = self._create_difference_mask(pixels1, pixels2)
        result.heatmap = self._create_heatmap(pixels1, pixels2)
        
        # Find difference regions
        result.difference_regions = self._find_difference_regions(
            result.difference_mask, pixels1, pixels2
        )
        
        # Calculate overall similarity
        result.overall_similarity = self._calculate_similarity(result.difference_mask)
        
        # Gather statistics
        result.stats = self._calculate_statistics(result)
        
        return result, pixels1, pixels2
    
    def _validate_compatibility(self, img1: ImageData, img2: ImageData) -> bool:
        """Check if images can be compared"""
        return (img1.metadata.width == img2.metadata.width and 
                img1.metadata.height == img2.metadata.height)
    
    def _create_difference_mask(self, pixels1: np.ndarray, pixels2: np.ndarray) -> np.ndarray:
        """Create binary mask showing where images differ"""
        # Ensure arrays have same shape
        assert pixels1.shape == pixels2.shape, f"Shape mismatch: {pixels1.shape} vs {pixels2.shape}"
        
        # Calculate absolute difference
        diff = np.abs(pixels1.astype(float) - pixels2.astype(float))
        
        # Apply tolerance threshold
        threshold = self.tolerance * 255
        
        if len(pixels1.shape) == 3 and pixels1.shape[2] > 1:  # Color image
            # Consider pixel different if any channel exceeds threshold
            mask = np.any(diff > threshold, axis=2)
        else:  # Grayscale or single channel
            if len(diff.shape) == 3:
                diff = diff[:, :, 0]  # Take first channel if somehow 3D
            mask = diff > threshold
            
        return mask.astype(np.uint8)
    
    def _create_heatmap(self, pixels1: np.ndarray, pixels2: np.ndarray) -> np.ndarray:
        """Create intensity map showing degree of difference"""
        # Ensure arrays have same shape
        assert pixels1.shape == pixels2.shape, f"Shape mismatch: {pixels1.shape} vs {pixels2.shape}"
        
        diff = np.abs(pixels1.astype(float) - pixels2.astype(float))
        
        if len(pixels1.shape) == 3 and pixels1.shape[2] > 1:  # Color image
            # Use maximum difference across channels
            heatmap = np.max(diff, axis=2)
        else:  # Grayscale or single channel
            if len(diff.shape) == 3:
                heatmap = diff[:, :, 0]  # Take first channel if somehow 3D
            else:
                heatmap = diff
            
        # Normalize to 0-255 range
        if np.max(heatmap) > 0:
            return (heatmap / np.max(heatmap) * 255).astype(np.uint8)
        else:
            return heatmap.astype(np.uint8)
    
    def _find_difference_regions(self, mask: np.ndarray, pixels1: np.ndarray, 
                               pixels2: np.ndarray) -> List[DifferenceRegion]:
        """Identify connected regions of differences"""
        from scipy import ndimage
        
        regions = []
        
        # Find connected components in the mask
        labeled_array, num_features = ndimage.label(mask)
        
        for region_id in range(1, num_features + 1):
            # Get region coordinates
            region_mask = (labeled_array == region_id)
            y_coords, x_coords = np.where(region_mask)
            
            if len(y_coords) == 0:
                continue
                
            # Calculate bounding box
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            
            # Calculate difference intensity
            if len(pixels1.shape) == 3:  # Color
                diff_values = np.abs(pixels1[region_mask].astype(float) - 
                                   pixels2[region_mask].astype(float))
                avg_diff = np.mean(np.max(diff_values, axis=1))
            else:  # Grayscale
                diff_values = np.abs(pixels1[region_mask].astype(float) - 
                                   pixels2[region_mask].astype(float))
                avg_diff = np.mean(diff_values)
            
            region = DifferenceRegion(
                x=int(x_min),
                y=int(y_min),
                width=int(x_max - x_min + 1),
                height=int(y_max - y_min + 1),
                difference_type=DifferenceType.PIXEL_CHANGE,
                confidence=min(avg_diff / 255.0, 1.0),
                pixel_count=int(np.sum(region_mask)),
                avg_intensity_diff=float(avg_diff)
            )
            regions.append(region)
        
        return regions
    
    def _calculate_similarity(self, mask: np.ndarray) -> float:
        """Calculate overall similarity score"""
        if mask.size == 0:
            return 1.0
        total_pixels = mask.size
        different_pixels = np.sum(mask)
        return 1.0 - (different_pixels / total_pixels)
    
    def _calculate_statistics(self, result: ComparisonResult) -> Dict[str, Any]:
        """Calculate various comparison statistics"""
        return {
            'total_pixels': result.difference_mask.size,
            'changed_pixels': np.sum(result.difference_mask),
            'change_percentage': np.sum(result.difference_mask) / result.difference_mask.size * 100,
            'num_regions': len(result.difference_regions),
            'largest_region_size': max([r.pixel_count for r in result.difference_regions], default=0)
        }

# Usage example and helper functions
class DifferenceVisualizer:
    """Helper class to visualize comparison results"""
    
    @staticmethod
    def create_overlay_image(original: np.ndarray, mask: np.ndarray, 
                           highlight_color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
        """Create overlay showing differences in color"""
        overlay = original.copy()
        if len(original.shape) == 3:  # Color image
            overlay[mask > 0] = highlight_color
        return overlay
    
    @staticmethod
    def create_side_by_side(img1: np.ndarray, img2: np.ndarray, 
                           result: ComparisonResult) -> np.ndarray:
        """Create side-by-side comparison with difference overlay"""
        height, width = img1.shape[:2]
        
        if len(img1.shape) == 3:  # Color images
            combined = np.zeros((height, width * 3, 3), dtype=np.uint8)
            combined[:, :width] = img1
            combined[:, width:width*2] = img2
            combined[:, width*2:] = DifferenceVisualizer.create_overlay_image(
                img1, result.difference_mask
            )
        
        return combined
    
    @staticmethod
    def point_out_differences(img1: np.ndarray, img2: np.ndarray, 
                            result: ComparisonResult, 
                            style: str = 'bounding_boxes') -> np.ndarray:
        """
        Create annotated images showing exactly where differences are
        
        Args:
            img1, img2: Original images
            result: ComparisonResult with difference regions
            style: 'bounding_boxes', 'circles', 'arrows', 'numbers', or 'heatmap'
        
        Returns:
            Annotated image with differences clearly marked
        """
        height, width = img1.shape[:2]
        channels = 3 if len(img1.shape) == 3 else 1
        
        # Create output image (side by side + annotations)
        if channels == 3:
            output = np.zeros((height, width * 2, 3), dtype=np.uint8)
            output[:, :width] = img1
            output[:, width:] = img2
        else:
            output = np.zeros((height, width * 2), dtype=np.uint8)
            output[:, :width] = img1
            output[:, width:] = img2
            
        # Convert to color if needed for annotations
        if channels == 1:
            output = np.stack([output, output, output], axis=2)
        
        if style == 'bounding_boxes':
            output = DifferenceVisualizer._draw_bounding_boxes(output, result, width)
        elif style == 'circles':
            output = DifferenceVisualizer._draw_circles(output, result, width)
        elif style == 'arrows':
            output = DifferenceVisualizer._draw_arrows(output, result, width)
        elif style == 'numbers':
            output = DifferenceVisualizer._draw_numbered_regions(output, result, width)
        elif style == 'heatmap':
            output = DifferenceVisualizer._create_heatmap_overlay(img1, img2, result)
            
        return output
    
    @staticmethod
    def _draw_bounding_boxes(output: np.ndarray, result: ComparisonResult, width_offset: int) -> np.ndarray:
        """Draw colored bounding boxes around difference regions"""
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                 (255, 0, 255), (0, 255, 255)]  # Different colors for each region
        
        for i, region in enumerate(result.difference_regions):
            color = colors[i % len(colors)]
            thickness = max(1, int(region.confidence * 5))  # Thicker box = bigger difference
            
            # Draw on both images
            for x_offset in [0, width_offset]:
                # Draw rectangle
                cv2.rectangle(output, 
                            (region.x + x_offset, region.y),
                            (region.x + region.width + x_offset, region.y + region.height),
                            color, thickness)
                
                # Add confidence label
                label = f"{region.confidence:.2f}"
                cv2.putText(output, label, 
                          (region.x + x_offset, region.y - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return output
    
    @staticmethod
    def _draw_circles(output: np.ndarray, result: ComparisonResult, width_offset: int) -> np.ndarray:
        """Draw circles around difference regions"""
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                 (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 255, 0)]
        
        for i, region in enumerate(result.difference_regions):
            color = colors[i % len(colors)]
            
            # Calculate circle center and radius based on region
            center_x = region.x + region.width // 2
            center_y = region.y + region.height // 2
            
            # Radius should encompass the entire region plus some padding
            radius = int(max(region.width, region.height) * 0.7) + 15
            
            # Adjust thickness based on confidence (higher confidence = thicker circle)
            thickness = max(2, int(region.confidence * 8))
            
            # Draw on both images
            for x_offset in [0, width_offset]:
                # Main circle
                cv2.circle(output, (center_x + x_offset, center_y), radius, color, thickness)
                
                # Optional: Add inner dotted circle for emphasis
                inner_radius = max(10, radius - 10)
                cv2.circle(output, (center_x + x_offset, center_y), inner_radius, color, 1)
                
                # Add region number with background
                text = str(i + 1)
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                
                # Background circle for number
                cv2.circle(output, (center_x + x_offset, center_y - radius - 25), 15, color, -1)
                
                # Number text
                cv2.putText(output, text, 
                          (center_x + x_offset - text_size[0]//2, center_y - radius - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Add confidence percentage below
                conf_text = f"{region.confidence:.1%}"
                cv2.putText(output, conf_text,
                          (center_x + x_offset - 20, center_y + radius + 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return output
    
    @staticmethod
    def draw_region_circles(img1: np.ndarray, img2: np.ndarray, 
                           result: ComparisonResult,
                           circle_color: Tuple[int, int, int] = (0, 255, 0),
                           thickness: int = 3,
                           show_numbers: bool = True,
                           show_confidence: bool = True,
                           padding: int = 20) -> np.ndarray:
        """
        Draw circles around detected difference regions with customization options
        
        Args:
            img1, img2: Original images
            result: ComparisonResult with difference regions
            circle_color: RGB color for circles (default: green)
            thickness: Circle line thickness
            show_numbers: Whether to number each region
            show_confidence: Whether to show confidence percentage
            padding: Extra pixels around region for circle size
            
        Returns:
            Side-by-side image with circles around differences
        """
        height, width = img1.shape[:2]
        channels = 3 if len(img1.shape) == 3 else 1
        
        # Create output image (side by side)
        if channels == 3:
            output = np.zeros((height, width * 2, 3), dtype=np.uint8)
            output[:, :width] = img1
            output[:, width:] = img2
        else:
            output = np.zeros((height, width * 2), dtype=np.uint8)
            output[:, :width] = img1
            output[:, width:] = img2
            # Convert to color for annotations
            output = np.stack([output, output, output], axis=2)
        
        # Draw circles around each region
        for i, region in enumerate(result.difference_regions):
            if region.confidence < 0.6:
                continue
            # Calculate circle parameters
            center_x = region.x + region.width // 2
            center_y = region.y + region.height // 2
            
            # Circle radius - encompasses the region with padding
            radius = int(max(region.width, region.height) / 2) + padding
            
            # Draw circle on both images
            for x_offset in [0, width]:
                cv2.circle(output, (center_x + x_offset, center_y), 
                          radius, circle_color, thickness)
                
                if show_numbers:
                    # Number the region
                    text = str(i + 1)
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                    
                    # Background for number
                    cv2.circle(output, (center_x + x_offset, center_y - radius - 30), 
                             18, circle_color, -1)
                    
                    # White number text
                    cv2.putText(output, text,
                              (center_x + x_offset - text_size[0]//2, 
                               center_y - radius - 25),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                
                if show_confidence:
                    # Show confidence below circle
                    conf_text = f"{region.confidence:.0%}"
                    cv2.putText(output, conf_text,
                              (center_x + x_offset - 15, center_y + radius + 25),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, circle_color, 2)
        
        return output
    
    @staticmethod
    def _draw_arrows(output: np.ndarray, result: ComparisonResult, width_offset: int) -> np.ndarray:
        """Draw arrows pointing from image1 to corresponding regions in image2"""
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        
        for i, region in enumerate(result.difference_regions):
            color = colors[i % len(colors)]
            
            # Arrow from region in img1 to same region in img2
            start_point = (region.x + region.width // 2, region.y + region.height // 2)
            end_point = (region.x + region.width // 2 + width_offset, region.y + region.height // 2)
            
            # Draw arrow
            cv2.arrowedLine(output, start_point, end_point, color, 3, tipLength=0.1)
            
            # Add change percentage
            midpoint_x = start_point[0] + (end_point[0] - start_point[0]) // 2
            midpoint_y = start_point[1] - 20
            change_pct = (region.pixel_count / (region.width * region.height)) * 100
            label = f"{change_pct:.1f}%"
            cv2.putText(output, label, (midpoint_x - 15, midpoint_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return output
    
    @staticmethod
    def _draw_numbered_regions(output: np.ndarray, result: ComparisonResult, width_offset: int) -> np.ndarray:
        """Number each difference region with detailed info"""
        for i, region in enumerate(result.difference_regions):
            color = (255, 255, 255)  # White text
            bg_color = (0, 0, 0)     # Black background
            
            # Draw on both images
            for x_offset in [0, width_offset]:
                # Draw number with background
                text = str(i + 1)
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                
                # Background rectangle
                cv2.rectangle(output,
                            (region.x + x_offset - 2, region.y - text_size[1] - 5),
                            (region.x + x_offset + text_size[0] + 2, region.y + 2),
                            bg_color, -1)
                
                # Number text
                cv2.putText(output, text,
                          (region.x + x_offset, region.y - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return output
    
    @staticmethod
    def _create_heatmap_overlay(img1: np.ndarray, img2: np.ndarray, result: ComparisonResult) -> np.ndarray:
        """Create a heatmap overlay showing intensity of differences"""
        import cv2
        
        # Create heatmap colormap
        heatmap_colored = cv2.applyColorMap(result.heatmap, cv2.COLORMAP_JET)
        
        # Blend with original image
        alpha = 0.6
        blended = cv2.addWeighted(img1, alpha, heatmap_colored, 1-alpha, 0)
        
        # Create side-by-side with original and heatmap overlay
        height, width = img1.shape[:2]
        output = np.zeros((height, width * 2, 3), dtype=np.uint8)
        output[:, :width] = img1
        output[:, width:] = blended
        
        return output

class ImageRender:
    def __init__(self, img1_path: str, img2_path: str):
        self.img1_path = img1_path
        self.img2_path = img2_path

    def get_img_url(self):
        
        img1 = ImageData(self.img1_path)
        img2 = ImageData(self.img2_path)
        
        # Create comparator
        comparator = ImageComparator(tolerance=0.05)

        # Perform comparison
        result, img1, img2 = comparator.compare(img1, img2)

        # Create visualizer
        visualizer = DifferenceVisualizer()

        circled_img = visualizer.draw_region_circles(
            img1, img2, result,
            circle_color=(255, 0, 0),  
            thickness=2,
            show_numbers=False,          # No numbers
            show_confidence=True        # No confidence text
        )
        # Encode image to memory buffer (e.g., JPG format)
        _, buffer = cv2.imencode('.jpg', circled_img)

        # Convert buffer to base64 string
        base64_image = base64.b64encode(buffer).decode('utf-8')

        # Optionally, prepend data URI for embedding in HTML or Dash
        data_url = f"data:image/jpg;base64,{base64_image}"
        
        return data_url

def export_images(schema1,schema2,img_name):

    # Since fetch_image_data_from_db returns memoryview objects, we need to convert them to PIL Images
    img1_data, ext1 = fetch_image_data_from_db(schema1, img_name)
    img2_data, ext2 = fetch_image_data_from_db(schema2, img_name)
    
    # Convert memoryview to PIL Image
    img1 = Image.open(io.BytesIO(img1_data))
    img2 = Image.open(io.BytesIO(img2_data))
    
    # Standardize the heights
    max_height = max(img1.height, img2.height)
    width = (img1.width + img2.width)//2
    img1 = img1.resize((width, max_height))
    img2 = img2.resize((width, max_height))

    # Create a new blank image with extra space for text
    text_height = 100  # Adjust height for text
    merged_img = Image.new('RGB', (img1.width + img2.width, img1.height + text_height), (255, 255, 255))

    # Paste images side by side
    merged_img.paste(img1, (0, text_height))
    merged_img.paste(img2, (img1.width, text_height))

    # Add text
    draw = ImageDraw.Draw(merged_img)
    font = ImageFont.truetype("arial.ttf", 60)
    text1 = "rev"+schema1.split("rev")[-1]+": "+img_name[:30]+"..."
    text2 = "rev"+schema2.split("rev")[-1]+": "+img_name[:30]+"..."

    # Get text bounding box (works in newer versions of Pillow)
    bbox1 = draw.textbbox((0, 0), text1, font=font)
    text_width1 = bbox1[2] - bbox1[0] 

    bbox2 = draw.textbbox((0, 0), text2, font=font)
    text_width1 = bbox2[2] - bbox2[0] 

    text_height = 20 

    draw.text(((img1.width - text_width1) // 2, 10), text1, font=font, fill=(0, 0, 0))

    draw.text(((img1.width - text_width1) // 2 + img1.width, 10), text2, font=font, fill=(0, 0, 0))

    return merged_img