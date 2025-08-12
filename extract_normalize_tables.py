import ftfy
import tempfile, os, shutil, io, re
import numpy as np, pandas as pd
import camelot, pdfplumber
from .processing_lines import *
from .pdf_management import *
from .database_management import *

#extract and normalize tables
def extract_raw_tables(pdf, wlinedf, start_page=0, end_page=0, flavor='stream', edge_tol=80)->list:
    '''
    Extract raw table with camelot because a lot of tables are borderless so other table extracting libs cant do
    Use hybird to extract borderless table and avoid joined col because of the 'stream' flavor
    raw table will contains table values and coordinates of cells in the table
    '''
   
    # get start and end page
    if start_page == 0:
        first_table_name_df = get_all_table_name(pdf)
        first_table_name_df['num'] = first_table_name_df['table name'].apply(lambda x: re.findall("\d+", x.split(" ")[1])[0] if len(re.findall("\d+", x.split(" ")[1]))>0 else 0)
        first_table_name = first_table_name_df[(first_table_name_df['num'] != 0) & \
                                                (first_table_name_df['table name'].apply(lambda x: len(x.split(" ")) > 2))]\
                                                .sort_values(by='page')
        start_page = first_table_name['page'].iloc[0]

    if end_page ==0:
        end_page = get_no_pages(pdf)

    # get raw tables
    table_list = []
    for i in range(start_page, end_page+1):
        table_areas, columns_delimiters = get_table_lines(i, wlinedf, pdf)
        tables = camelot.read_pdf(pdf, flavor= flavor, 
                                        pages=f'{i}', 
                                        layout_kwargs={'boxes_flow': 0.5, 'line_overlap': 0.5}, 
                                        edge_tol=edge_tol,
                                        row_tol = 10,
                                        parallel=True,
                                        table_areas=table_areas,
                                        columns=columns_delimiters,                                        
                                        suppress_stdout=True)

        for raw_table in tables:   
            try:            
                table = raw_table.df
                if table.empty or len(table) <2:
                    continue
                table['page'] = i
                cols_to_drop = []
                for col in range(table.shape[1]):
                    if all(cell == "" for cell in table.iloc[:,col].tolist()):
                        cols_to_drop.append(col)
                table = table.drop(columns = cols_to_drop, axis =1)
                columns = [col for col in table.columns if col not in cols_to_drop]
                coordinates = [[(round(cell.x1,2), round(cell.y1, 2), round(cell.x2, 2), round(cell.y2,2)) for col_no, cell in enumerate(row) if col_no in columns] for row in raw_table.cells]         
                table['coordinates'] = pd.Series(coordinates)
                refined_table = refine_words(table) #avoid ligatures
                table_list.append(refined_table)
            except Exception as e:
                print(e)           
    return table_list

def extract_table_RDD(pdf_path):
    pages = get_no_pages(pdf_path)
    table_list = []
    for i in range(5, pages):
        tables = camelot.read_pdf(pdf_path, pages = f'{i}', split_text=True)
        for raw_table in tables:
            table = raw_table.df
            table = table.apply(lambda x: x.replace("", np.nan))
            if not table.isna().all().all():
                table = table.fillna("")
                table['page'] = i
                columns = table.columns
                coordinates = [[(round(cell.x1,2), round(cell.y1, 2), round(cell.x2, 2), round(cell.y2,2)) for col_no, cell in enumerate(row) if col_no in columns] for row in raw_table.cells]         
                table['bbox'] = pd.Series(coordinates)
                refined_table = refine_words(table)
                refined_table['contents'] = assign_contents(refined_table, table_of_contents)
                standardized_table = standardize_header(refined_table)
                table_list.append(standardized_table)
    assigned_table = assign_table_name(table_list)   
    merged_table_list = merge_tables(assigned_table)
    transformed_table_list = transform_tables(merged_table_list)
    return transformed_table_list

def split_tables(table)-> list:
    '''
    split joined tables because on top of each table will have table name, based on that table name we could split them
    '''
    #get index of table names to split
    matches = table.apply(lambda row: row.str.contains(r"^Table |^Relay Lens Set-Up Requirements").any(), axis=1)
    indexes = table[matches].index.tolist()
    splited_tables = []
    if len(indexes) > 0:
        for i in range(len(indexes)):
            table_name = [cell for cell in table.iloc[indexes[i]].tolist() if (cell !="") and (cell != "\\n")][0]

            #if only 1 table name in that table or last table name
            if indexes[i] == indexes[-1]:
                splited_table = table.iloc[indexes[i]+1:].reset_index(drop=True)
            
            #other tables between table names
            elif i < len(indexes) - 1:
                splited_table = table.iloc[indexes[i]+1:indexes[i+1]].reset_index(drop=True)

            #the rest of the splitted table, it might contain a table -> could be deleted in the clean table function
            else:
                splited_table = table.iloc[:indexes[i]].reset_index(drop=True)

            splited_table["Title"] = table_name
            if not splited_table.empty and len(splited_table)>=2:
                splited_tables.append(splited_table)
    else:
         splited_tables.append(table)
    return splited_tables

def clean_table(table, wlinedf):
    '''
    based on the line of table we could identify whether that row is belong to the table or not
    wlinedf: all the lines in the pdf and will be global
    '''
    linedf = wlinedf[(wlinedf['page']==table.iloc[0]['page']) & (wlinedf['type'] == 'vertical')]

    #handle exception
    pattern_close = r"\)"
    pattern_open = r"\("
    
    pattern_close = re.compile(pattern_close)
    pattern_open = re.compile(pattern_open)
    
    matches_oc = table.map(lambda x: bool(pattern_close.search(str(x))) and not bool(pattern_open.search(str(x))))
    cors_oc = np.where(matches_oc.values)

    #remove text outside tables
    rows_to_drop = []
    for i in range(len(table)):
        top_lines = table.iloc[i]['coordinates'][0][3]
        bottom_lines = table.iloc[i]['coordinates'][0][1]
        middle = (top_lines+bottom_lines)/2
        cell_size = (top_lines-bottom_lines)
        if cell_size > 25:
            filtered_linedf = linedf.loc[(linedf['y1'] > middle-cell_size*0.15) & (linedf['y0'] < middle+cell_size*0.15)]
        else:
            filtered_linedf = linedf.loc[(linedf['y1'] > middle) & (linedf['y0'] < middle)]

        if len(filtered_linedf) == 0:
            if i in cors_oc[0].tolist():#handle exception
                continue 
            rows_to_drop.append(i)
    #drop rows that not in any vertical lines
    rows_to_drop = [i for i in rows_to_drop if i in table.index]
    table = table.drop(rows_to_drop, axis = 0).reset_index(drop=True)

    #remove blank columns
    mask = ~(table == "").all(axis=0)
    table = table.loc[:,mask]
    return table

def assign_contents(table, table_of_contents):
    if "bbox" not in table.columns:
        return None
    chapters = table_of_contents[(table_of_contents['page'] < table['page'].iloc[0]) | \
                                                    ((table_of_contents['page'] == table['page'].iloc[0]) & (table_of_contents['position'] >= table['bbox'].iloc[1][0][1]))]
    chapter = chapters.iloc[-1]
    return chapter['contents']

def identify_bbox(table, wlinedf):
    """
    Optimized version of identify_bbox with vectorized operations
    Performance improvements:
    - Pre-filter lines once per page instead of per cell
    - Vectorized coordinate calculations
    - Cached line lookups
    - Reduced DataFrame operations
    """
    if table.empty:
        return table
        
    page_num = table.iloc[0]['page']
    
    # Pre-filter and cache lines for this page (MAJOR OPTIMIZATION)
    page_lines = wlinedf[wlinedf['page'] == page_num].copy()
    if page_lines.empty:
        return table
        
    vertical_lines = page_lines[page_lines['type'] == 'vertical'].sort_values('x0').reset_index(drop=True)
    horizontal_lines = page_lines[page_lines['type'] == 'horizontal'].sort_values('y0', ascending=False).reset_index(drop=True)
    
    # Pre-compute line boundaries for faster lookup
    v_x_coords = vertical_lines['x0'].values
    v_y0_coords = vertical_lines['y0'].values  
    v_y1_coords = vertical_lines['y1'].values
    
    h_x0_coords = horizontal_lines['x0'].values
    h_x1_coords = horizontal_lines['x1'].values
    h_y_coords = horizontal_lines['y0'].values
    
    # Get table boundaries once
    if not vertical_lines.empty and not horizontal_lines.empty:
        min_x0 = h_x0_coords.min()
        max_x1 = h_x1_coords.max() 
        min_y0 = v_y0_coords.min()
        max_y1 = v_y1_coords.max()
    else:
        # Fallback to original coordinates
        all_coords = []
        for _, row in table.iterrows():
            all_coords.extend(row['coordinates'])
        if all_coords:
            xs = [c[0] for c in all_coords] + [c[2] for c in all_coords]
            ys = [c[1] for c in all_coords] + [c[3] for c in all_coords]
            min_x0, max_x1 = min(xs), max(xs)
            min_y0, max_y1 = min(ys), max(ys)
        else:
            return table
    
    optimized_bbox_list = []
    
    for row_idx, row in table.iterrows():
        coords = row['coordinates']
        if not coords:
            optimized_bbox_list.append([])
            continue
            
        # Vectorized coordinate processing
        coords_array = np.array(coords)
        x0_vals = coords_array[:, 0]
        y0_vals = coords_array[:, 1] 
        x1_vals = coords_array[:, 2]
        y1_vals = coords_array[:, 3]
        
        # Calculate cell properties vectorized
        y_centers = (y0_vals + y1_vals) / 2
        cell_heights = y1_vals - y0_vals
        
        row_bbox = []
        
        for cell_idx in range(len(coords)):
            cell_coord = coords[cell_idx]
            y_center = y_centers[cell_idx]
            cell_height = cell_heights[cell_idx]
            
            # Optimized line filtering using numpy
            if cell_height < 20:
                if row_idx == 0:
                    tolerance = cell_height * 0.3
                    v_mask = (v_y0_coords <= y0_vals[cell_idx] + tolerance) & \
                            (v_y1_coords >= y1_vals[cell_idx] - tolerance)
                else:
                    v_mask = (v_y0_coords <= y_center) & (v_y1_coords >= y_center)
            else:
                tolerance = cell_height * 0.15
                v_mask = (v_y0_coords <= y_center + tolerance) & \
                        (v_y1_coords >= y_center - tolerance)
            
            # Get relevant lines
            relevant_v_lines = vertical_lines[v_mask] if v_mask.any() else pd.DataFrame()
            
            h_mask = (h_x0_coords <= cell_coord[0] + 6) & \
                    (h_x1_coords >= cell_coord[2] - 6)
            relevant_h_lines = horizontal_lines[h_mask] if h_mask.any() else pd.DataFrame()
            
            # Skip if no lines found and not first row
            if relevant_v_lines.empty and row_idx != 0:
                if row_idx > 0 and cell_idx < len(optimized_bbox_list[row_idx-1]):
                    row_bbox.append(optimized_bbox_list[row_idx-1][cell_idx])
                else:
                    row_bbox.append(tuple(cell_coord))
                continue
            
            # Calculate new coordinates efficiently
            new_coord = list(cell_coord)
            
            # X-coordinates (vertical lines)
            if not relevant_v_lines.empty:
                right_v_lines = relevant_v_lines[relevant_v_lines['x0'] >= cell_coord[0] - 6]
                if not right_v_lines.empty:
                    new_coord[0] = right_v_lines.iloc[0]['x0']
                else:
                    new_coord[0] = max_x1
                    
                right_v_lines_end = relevant_v_lines[relevant_v_lines['x0'] >= cell_coord[2] - 6]
                if not right_v_lines_end.empty:
                    new_coord[2] = right_v_lines_end.iloc[0]['x0']
                else:
                    new_coord[2] = max_x1
            
            # Y-coordinates (horizontal lines)  
            if not relevant_h_lines.empty:
                y_threshold = y_center if cell_height <= 30 else cell_coord[1] + cell_height * 0.3
                
                under_h_lines = relevant_h_lines[
                    (relevant_h_lines['y0'] <= y_threshold) & 
                    (relevant_h_lines['y0'] >= min_y0)
                ]
                if not under_h_lines.empty:
                    new_coord[1] = under_h_lines.iloc[0]['y0']
                else:
                    new_coord[1] = min_y0
                    
                above_h_lines = relevant_h_lines[
                    (relevant_h_lines['y0'] > y_threshold) & 
                    (relevant_h_lines['y0'] <= max_y1 + 16)
                ].sort_values('y0', ascending=False)
                if not above_h_lines.empty:
                    new_coord[3] = above_h_lines.iloc[-1]['y0']
                else:
                    new_coord[3] = max_y1
            
            # Handle edge cases
            if new_coord[0] == new_coord[2]:
                if cell_idx == 0:
                    new_coord[0] = cell_coord[0]
                elif row_bbox:
                    new_coord[0] = row_bbox[-1][0]
                    
            if new_coord[1] == new_coord[3]:
                new_coord[3] = max_y1
            
            row_bbox.append(tuple(new_coord))
        
        optimized_bbox_list.append(row_bbox)
    
    # Update table
    table_copy = table.copy()
    table_copy['bbox'] = optimized_bbox_list
    table_copy.drop(columns=['coordinates'], inplace=True, errors='ignore')
    
    return table_copy

def merge_cells(table):
    """
    Optimized version of merge_cells with reduced DataFrame operations
    Performance improvements:
    - Vectorized bbox operations
    - Efficient duplicate detection
    - Optimized grouping and merging
    - Reduced string operations
    """
    if table.empty or "bbox" not in table.columns:
        return table

    contents_end = table.columns.tolist().index('page')
    
    # Extract content and bbox data efficiently
    content_cols = table.columns[:contents_end]
    content_data = table[content_cols].values
    bbox_data = table['bbox'].values
    
    # Flatten data for processing - more efficient than melting
    flattened_content = []
    flattened_bbox = []
    row_indices = []
    col_indices = []
    
    for row_idx, (content_row, bbox_row) in enumerate(zip(content_data, bbox_data)):
        for col_idx, (content_val, bbox_val) in enumerate(zip(content_row, bbox_row)):
            if content_val == None:
                content_val = ""
            flattened_content.append(str(content_val).strip())
            flattened_bbox.append(tuple(bbox_val) if isinstance(bbox_val, (list, tuple)) else bbox_val)
            row_indices.append(row_idx)
            col_indices.append(col_idx)
    
    if not flattened_bbox:
        return table
    
    # Create mapping for efficient grouping
    bbox_to_content = {}
    bbox_to_positions = {}
    
    for content, bbox, row_idx, col_idx in zip(flattened_content, flattened_bbox, row_indices, col_indices):
        if bbox not in bbox_to_content:
            bbox_to_content[bbox] = []
            bbox_to_positions[bbox] = []
        bbox_to_content[bbox].append(content)
        bbox_to_positions[bbox].append((row_idx, col_idx))
    
    # Identify merged cells (bbox appearing multiple times)
    merged_bbox = {bbox: contents for bbox, contents in bbox_to_content.items() if len(contents) > 1}
    if not merged_bbox:
        # No merged cells, return original table with minimal processing
        result_table = table.copy()
        result_table['bbox'] = result_table['bbox'].apply(lambda x: [tuple(b) if isinstance(b, (list, tuple)) else b for b in x])
        return result_table
    
    # Merge content for cells with same bbox - optimized
    merged_content = {}
    for bbox, content_list in merged_bbox.items():
        # Apply normalize_sentence to joined content
        joined_content = ' '.join(content_list).strip()
        if joined_content:
            try:
                normalized = normalize_sentence(joined_content)
                merged_content[bbox] = normalized
            except:
                merged_content[bbox] = joined_content
    
    # Update table with merged content efficiently
    result_table = table.copy()
    bbox_array = np.array(result_table['bbox'].tolist(), dtype=object)
    
    for row_idx in range(len(result_table)):
        for col_idx in range(contents_end):
            if row_idx < bbox_array.shape[0] and col_idx < bbox_array.shape[1]:
                cell_bbox = tuple(bbox_array[row_idx, col_idx]) if bbox_array[row_idx, col_idx].any() else None
                if cell_bbox in merged_content:
                    result_table.iloc[row_idx, col_idx] = merged_content[cell_bbox]
    
    # Remove duplicates efficiently
    result_table['bbox'] = result_table['bbox'].apply(lambda x: tuple(tuple(b) if isinstance(b, (list, tuple)) else b for b in x))
    
    # Optimized duplicate removal
    duplicate_mask = result_table.duplicated()
    if duplicate_mask.any():
        result_table = result_table[~duplicate_mask].reset_index(drop=True)
    
    # Remove duplicate columns efficiently
    content_columns = [col for col in result_table.columns[:contents_end] if col != 'page']
    if content_columns:
        # Check for duplicate columns using numpy for speed
        unique_df = result_table[content_columns].T.drop_duplicates()
        unique_columns = list(unique_df.index)
        
        # Reconstruct table with unique columns
        other_columns = [col for col in result_table.columns if col not in content_columns]
        result_table = result_table[unique_columns + other_columns]
    
    # Clean up text formatting
    for col in content_columns:
        if col in result_table.columns:
            result_table[col] = result_table[col].astype(str).str.replace(r'\s*([^\w\d\s])\s*', r'\1', regex=True)
    
    # Handle exceptions efficiently with vectorized operations
    handle_text_exceptions_vectorized(result_table, content_columns)
    
    # Clean text patterns
    for col in content_columns:
        if col in result_table.columns:
            # Vectorized text cleaning
            mask = result_table[col].str.contains(r'(\w)\s*-\s*(\w)', regex=True, na=False) & \
                   ~result_table[col].str.contains('E-', na=False)
            result_table.loc[mask, col] = result_table.loc[mask, col].str.replace(r'(\w)\s*-\s*(\w)', r'\1\2', regex=True)
    
    result_table['page'] = result_table['page'].astype(int)
    return result_table

def handle_text_exceptions_vectorized(table, content_columns):
    """Handle text exceptions using vectorized operations"""
    
    # Pattern 1: Handle closing parenthesis exceptions
    pattern_exception = re.compile(r"^(?<!\()\w.*\)\n.")
    for col in content_columns:
        if col not in table.columns:
            continue
            
        matches = table[col].astype(str).str.contains(pattern_exception, regex=True, na=False)
        if matches.any():
            match_indices = matches[matches].index
            for idx in match_indices:
                if idx > 0:  # Can't process first row
                    content = table.at[idx, col]
                    if isinstance(content, str) and ')' in content:
                        parts = content.split(')', 1)
                        if len(parts) == 2:
                            # Update previous row
                            prev_content = str(table.at[idx-1, col])
                            table.at[idx-1, col] = f"{prev_content} {parts[0].strip()})"
                            
                            # Update current row
                            new_content = parts[1]
                            if '(' in new_content:
                                new_content += ')'
                            table.at[idx, col] = new_content
                            
                            # Clean up hyphen
                            table.at[idx-1, col] = str(table.at[idx-1, col]).replace('- ', '')

    # Pattern 2: Handle trailing hyphens/slashes
    patterns = [re.compile(r"\w+-$"), re.compile(r'\w+/$')]
    for col in content_columns:
        if col not in table.columns:
            continue
            
        for pattern in patterns:
            matches = table[col].astype(str).str.contains(pattern, regex=True, na=False)
            if matches.any():
                match_indices = matches[matches].index
                for idx in match_indices:
                    if idx < len(table) - 1:  # Not last row
                        current_content = str(table.at[idx, col])
                        next_content = str(table.at[idx+1, col])
                        
                        if '\n' in next_content:
                            parts = next_content.split('\n', 1)
                            # Merge with current
                            table.at[idx, col] = f"{current_content} {parts[0].strip()}"
                            # Update next row
                            table.at[idx+1, col] = ' '.join(parts[1:]).strip() if len(parts) > 1 else ''

    # Pattern 3: Remove notes at end
    note_patterns = [r"^Note \d:", r"^Notes:", r"^Note:"]
    combined_pattern = re.compile("|".join(note_patterns))
    
    for col in content_columns:
        if col not in table.columns:
            continue
            
        # Check only last row for notes
        if len(table) > 0:
            last_row_content = str(table.iloc[-1][col])
            if combined_pattern.search(last_row_content):
                # Remove last row
                table.drop(table.index[-1], inplace=True)
                break

def standardize_header(table):
    contents_end = table.columns.tolist().index('page')-1
    if table.iloc[0].str.contains(r",", regex=True).any() or table.iloc[0].str.contains(r":$", regex=True).any():
        return table

    def handle_exceptional_headers(row):
        for i in range(row):
            if (table.iloc[i,0] == table.iloc[i,1]):
                headers = re.split(r"(?<! ) (?! )", table.iloc[i,0])
                if len(headers) > 1:
                    table.iloc[i,1] = headers[1]
                    table.iloc[i,0] = headers[0]

            if table.iloc[i,contents_end-1] == table.iloc[i,contents_end]:
                headers = re.split(r"(?<! ) (?! )", table.iloc[i,contents_end-1])
                if len(headers) > 1:
                    table.iloc[i,contents_end] = headers[1]
                    table.iloc[i,contents_end-1] = headers[0]

    if ((table.iloc[0,0] == table.iloc[1,0]) and (table.iloc[0,0] != "")) \
        or (table.iloc[0,contents_end-1] == table.iloc[1,contents_end-1] and (table.iloc[0,contents_end-1] != "")) \
        or ((table.iloc[1,0]=="") and (table.iloc[1,1]=="") and (table.iloc[0,2] == table.iloc[0,3])):

        if len(table) > 2 and ((table.iloc[0,0] == table.iloc[2,0]) or (table.iloc[0,contents_end-1] == table.iloc[2,contents_end-1])):
            table = table.drop(1).reset_index(drop=True)

        handle_exceptional_headers(2)
        
        row0_c = [re.sub(r"\s+", " ", col.replace("\n", " ")).strip() for col in table.iloc[0, :contents_end+1].tolist()]
        row1_c = [re.sub(r"\s+", " ", col.replace("\n", " ")).strip() for col in table.iloc[1, :contents_end+1].tolist()]

        if row0_c.count("") >= 2:
            return table

        if table.iloc[1].tolist().count("")>2: #handle exception: merged column
            standardize_sub_headers = [sub if sub == row0_c[col] else char for col, sub in enumerate(row1_c) for char in (sub.split() if sub != row0_c[col] else [sub])]
            if len(standardize_sub_headers) == len(row0_c):
                row1_c = standardize_sub_headers

        for cell in range(0,contents_end+1):
            if row0_c[cell] == row1_c[cell]:
                continue
            else:
                row0_c[cell] = (row0_c[cell]+" "+row1_c[cell]).strip()

        table = table.drop(1).reset_index(drop=True)  

    else:
        handle_exceptional_headers(1)

        row0_c = [re.sub(r"\s+", " ", col.replace("\n", " ")).strip() for col in table.iloc[0, :contents_end+1].tolist()]
        row1_c = [re.sub(r"\s+", " ", col.replace("\n", " ")).strip() for col in table.iloc[1, :contents_end+1].tolist()]

        if row0_c.count("") >= 2:
            row0_c = row1_c
            table = table.drop(0).reset_index(drop=True)

        
    table.columns = row0_c + table.columns[contents_end+1:].tolist()
    table = table.drop(0).reset_index(drop=True)

    #handle exception
    headers = table.columns.tolist()
    for col_no, header in enumerate(headers):
        if "Parameter" in str(header):
            headers[col_no] = "Parameter"
    table.columns = headers
    return table

def assign_table_name(tables, tables_name)->list:
    
    if 'bbox' not in tables[0].columns:
        return tables
    assigned_tables = []
    for table in tables:
        if table.empty:
            continue
        if 'Title' in table.columns:
            assigned_tables.append(table)
            continue
        table_name_list = tables_name[
    (
        ((tables_name['page'] == table.iloc[0]['page']) & (abs(tables_name['top'] - table['bbox'].iloc[0][0][3]) < 100))
        &
        ((tables_name['top']-table['bbox'].iloc[-1][0][1])>0)
    ) |
    (
        (tables_name['page'] == table.iloc[0]['page'] - 1) &
        (tables_name['top'] <= 150)
    )]
        if not table_name_list.empty:
            table_name = table_name_list['table name'].iloc[-1]
            table['Title'] = table_name
        assigned_tables.append(table)
    return assigned_tables

def fillna_span_parameter(table_tail, table_i):
    if table_tail.iloc[-1,0] =="" and table_i.iloc[0,0] !="":
        for i in range(-1, -len(table_tail)-1,-1):
            if table_tail.iloc[i,0] != "":
                break
            else:
                table_tail.iloc[i,0] = table_i.iloc[0,0]
    if table_tail.iloc[-1,0] !="" and table_i.iloc[0,0] =="":
        for i in range(0, len(table_i)):
            if table_i.iloc[i,0]!="":
                break
            else:
                table_i.iloc[i,0] = table_tail.iloc[-1,0]

    return (table_tail, table_i)

def join_tables(table_tail, table_i):
    #avoid missing first column
    if abs(table_tail['bbox'].iloc[0][0][0] - table_i['bbox'].iloc[0][0][0]) <10:
        table_tail, table_i = fillna_span_parameter(table_tail, table_i)
        
    #merge tables where tables have same columns name
    if all(header in table_tail.columns.tolist() for header in table_i.columns.tolist()):
        table_tail = table_tail.reset_index(drop=True)
        table_i = table_i.reset_index(drop=True)
        table_tail = pd.concat([table_tail, table_i], axis = 0, ignore_index = True, sort=False)

    #merge table with different columns name due to missing data at header
    elif len(table_tail.columns.tolist())-1 == len(table_i.columns.tolist()):
        table_i.insert(table_tail.columns.tolist().index("Title"),"Title","")     
        headers = table_tail.columns.tolist()
        merged_values = np.concatenate((table_tail.values, table_i.values), axis=0)
        table_tail = pd.DataFrame(merged_values, columns = headers)
    
    #merge table with different columns name due to no headers at origin table
    elif len(table_tail.columns.tolist())-1 >= len(table_i.columns.tolist()) and (any(isinstance(header, int) for header in table_i.columns.tolist())):
        #find row without merged cells
        headers = table_tail.columns.tolist()
        #get vertical lines to split merged columns into smaller columns
        #find row with greatest number of columns
        contents_end_table_tail = table_tail.columns.tolist().index('page')
        vlines_table_tail = set()
        for r in range(len(table_tail)):
            vlines_table_tail = list(set([bbox[0] for bbox in table_tail.iloc[r]['bbox']] + [table_tail.iloc[r]['bbox'][-1][3]]))
            if len(vlines_table_tail) == contents_end_table_tail+1:
                break
        #find row with greatest number of columns
        contents_end_table_i = table_i.columns.tolist().index('page')
        vlines_table_i = set()
        for r in range(len(table_i)):
            vlines_table_i= list(set([bbox[0] for bbox in table_i.iloc[r]['bbox']] + [table_i.iloc[r]['bbox'][-1][3]]))
            if len(vlines_table_i) == contents_end_table_i+1:
                break
        
        vlines_table_tail.sort()
        vlines_table_i.sort()
        #insert and fill to split merged columns
        for col, lines in enumerate(vlines_table_tail):
            if not any(abs(lines - linesi) <= 3 for linesi in vlines_table_i):
                if col==0:
                    col_contents = " "
                else:
                    col_contents = pd.Series(table_i.iloc[:,col-1])
                table_i.insert(col, headers[col],col_contents)
        #merge table
        table_i.insert(table_tail.columns.tolist().index("Title"),"Title","")               
        merged_values = np.concatenate((table_tail.values, table_i.values), axis=0)
        table_tail = pd.DataFrame(merged_values, columns = headers)

    table_tail.fillna("")
    return table_tail

def merge_tables(table_list)->list:
    
    if len(table_list)<2:
        return table_list
    merged_table_list = []
    k = 0
    for i, table_i in enumerate(table_list):
        if i==k:
            merged_table_list.append(table_i)
            continue

        table_i = table_i.reset_index(drop=True)
        # print(tabulate(table_i, table_i.columns, "grid"))
        table_tail = merged_table_list[-1].reset_index(drop=True)

        table_i = table_i.rename(columns=lambda x: str(x).replace("  ", " "))
        table_tail = table_tail.rename(columns=lambda x: str(x).replace("  ", " "))
        content_end = table_tail.columns.tolist().index('page')
        try:
            if ('Title' in table_tail.columns) and ('Title' in table_i.columns):
                print("Title in both tables")
                if (table_i['Title'].iloc[0] == table_tail['Title'].iloc[0]):
                    table_tail = join_tables(table_tail, table_i)
                    merged_table_list[-1] = table_tail
                else:
                    merged_table_list.append(table_i)

            elif int(table_i['page'].iloc[0])-1 == int(table_tail['page'].iloc[-1]) and ((table_i['bbox'].iloc[0][0][3]>=650) and (table_tail['bbox'].iloc[-1][0][1]<=140)):
                #table tail and table i have no Title
                if ('Title' not in table_tail.columns) and ('Title' not in table_i.columns) and \
                    (table_tail.iloc[0]['contents'] == table_i.iloc[0]['contents']):
                    print("No Title in both tables")
                    if table_tail.columns.tolist() == table_i.columns.tolist():
                        table_tail = join_tables(table_tail, table_i)
                        merged_table_list[-1] = table_tail
                    else:
                        merged_table_list.append(table_i)
                #table tail has no Title
                elif ('Title' not in table_tail.columns) and ('Title' in table_i.columns): 
                    print("Table tail has no Title")
                    merged_table_list.append(table_i)
                #table tail has Title
                elif ('Title' in table_tail.columns) and ('Title' not in table_i.columns):
                    print("Table tail has Title")
                    if len(table_i.columns)+1 != len(table_tail.columns) and not any(isinstance(header, int) for header in table_i.columns.tolist()):
                        merged_table_list.append(table_i)
                        continue
                    table_tail = join_tables(table_tail, table_i)
                    merged_table_list[-1] = table_tail
                else:
                    merged_table_list.append(table_i)
                    print("Table tail and table i have Title")

            elif (table_tail.columns.tolist()[:content_end]) == (table_i.columns.tolist()[:content_end]):
                print(1)
                table_tail = join_tables(table_tail, table_i)
                merged_table_list[-1] = table_tail

            elif ('Title' not in table_tail.columns) or ('Title' not in table_i.columns):
                print(2)
                merged_table_list.append(table_i)

            else:
                print(3)
                merged_table_list.append(table_i)
        except pd.errors.InvalidIndexError as e:
            print(e)
            print(tabulate(table_tail, table_tail.columns, "grid"))
            print(tabulate(table_i, table_i.columns, "grid"))
            headers = table_tail.columns.tolist()
            merged_values = np.concatenate((table_tail.values, table_i.values), axis=0)
            table_tail = pd.DataFrame(merged_values, columns = headers)
            merged_table_list[-1] = table_tail
        except Exception as e:
            print(e)
            print(tabulate(table_tail))
            print(tabulate(table_i))
            merged_table_list.append(table_i)
        if 'Title' in merged_table_list[-1].columns.tolist():
            merged_table_list[-1]['Title'] = merged_table_list[-1].iloc[0]['Title']

    merged_table_list = [table.replace(np.nan, '', regex=True) for table in merged_table_list]

    return merged_table_list

def reshape_table(table):
    # table = table.loc[:, ~table.T.duplicated()]
    cols = table.columns.tolist()
    headers = [str(header) + str(cols[:col].count(header)) if cols.count(header)>1 else header for col, header in enumerate(cols)]
    table.columns = headers
    row_index = table.index.tolist()
    table.loc[:,'row_index'] = pd.Series(row_index)
    k=2
    if any(re.search(r"P\s*?a\s*?r\s*?a\s*?m\s*?e\s*?t\s*?e\s*?r", str(header)) for header in headers) or (any(re.search(r"M\s*?o\s*?d\s*?e\s*?", str(header)) for header in headers) and any(re.search(r"T\s*?r\s*?e\s*?n\s*?d\s*?N\s*?a\s*?m\s*?e", str(header)) for header in headers)):
        k+=1
    if 'Title' in headers:
        k+=1
    id_vars = headers[:k] + ['row_index']
    value_vars = [header for header in headers if header not in id_vars]
    melted_table = pd.melt(table, id_vars=id_vars, value_vars=value_vars, var_name='indicator', value_name='spec_value').sort_values(by=id_vars)
    melted_table['indicator'] = pd.Categorical(melted_table['indicator'], categories=value_vars, ordered=True)
    sorted_melted_table = melted_table.sort_values(by=['row_index', 'indicator']).drop(columns='row_index').reset_index(drop=True)
    return sorted_melted_table

def transform_tables(table_list):
    transformed_table_list = []
    for table in table_list:
        table = table.drop(columns=['bbox'])
        headers = table.columns.tolist()
        if headers.count("")>1:
            contents_end = headers.index('page')
            headers = table.iloc[0,:contents_end].tolist() + headers[contents_end:]
            table.columns = headers
            table = table.drop(0).reset_index(drop=True)

        new_headers = ['contents'] + [header for header in headers if header not in ['contents']]
        if 'Title' in table.columns.tolist():
            table['Title'] = table['Title'].str.replace(":", ".")
            table['Title'] = table['Title'].apply(lambda x: x.split(" Rev ")[0] if " Rev " in x else x)
            table['Title'] = table['Title'].apply(lambda x: (" ".join(x.split(" ")[2:])+".").strip(" ") if x.count(" ")>=2 else (" ".join(x.split(" ")[1:])+".").strip(" "))
            new_headers = ['contents', 'Title'] + [header for header in headers if header not in ['contents', 'Title']]
        transformed_table = table[new_headers]
        reshaped_table = reshape_table(transformed_table)
        transformed_table_list.append(reshaped_table)
    return transformed_table_list

def normalize_tables(table_list, wlinedf, pages_with_lines, table_of_contents, tables_name):
    """
    Optimized version of normalize_tables with performance improvements
    """
    print("Splitting tables...")
    splitted_table_list = []
    for table in table_list:
        splitted_table_list.extend(split_tables(table))
    
    print("Standardizing tables...")
    unique_table_list = []
    
    # Process pages in batches for memory efficiency
    pages_with_lines.sort()
    
    for page in pages_with_lines:
        tables_for_page = [table for table in splitted_table_list if int(table['page'].iloc[0]) == page + 1]
        
        if not tables_for_page:
            continue
            
        print(f"Processing page {page + 1}...")
        
        # Pre-filter lines for this page once
        
        standardized_tables = []
        for table in tables_for_page:
            # Clean table
            cleaned_table = clean_table(table, wlinedf)
            if cleaned_table.empty or len(cleaned_table) < 2:
                continue
            
            # Optimized bbox identification
            bboxed_table = identify_bbox(cleaned_table, wlinedf)
            bboxed_table['contents'] = assign_contents(bboxed_table, table_of_contents)
            
            # Optimized cell merging
            merged_table = merge_cells(bboxed_table)
            if len(merged_table) < 2:
                continue
            
            # Standardize headers
            standardized_table = standardize_header(merged_table)
            standardized_tables.append(standardized_table)
                
        unique_table_list.extend(standardized_tables)
    
    # Assign table names and merge
    print("Assigning table names and merging...")
    assigned_tables = assign_table_name(unique_table_list, tables_name)
    merged_table_list = merge_tables(assigned_tables)
    
    return merged_table_list