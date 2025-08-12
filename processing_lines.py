import pandas as pd
import pdfplumber
import math
from tabulate import tabulate
from .pdf_management import get_no_pages, get_page_info

#lines processing to identify tables layout
def preprocessing_lines(raw_linedf):
    raw_linedf = raw_linedf[['x0', 'y0', 'x1', 'y1', 'width', 'height', 'page']]
    linedf = raw_linedf.drop_duplicates().reset_index(drop=True)
    rows_to_drop = []
    for index in range(len(linedf)):
        if linedf.iloc[index]['width'] == 0:  #preprocessing vertical lines
            vlonger_lines = linedf[(linedf['x0']==linedf.iloc[index]['x0']) & (linedf['y0'] <= linedf.iloc[index]['y0']) & (linedf['y1'] >= linedf.iloc[index]['y1']) & (linedf.index != index)]
            if len(vlonger_lines)>0:
                rows_to_drop.append(index)

        elif linedf.iloc[index]['height'] == 0:  #preprocessing horizontal lines
            hlonger_lines = linedf[(linedf['y0'] == linedf.iloc[index]['y0']) & (linedf['x0'] <= linedf.iloc[index]['x0']) & (linedf['x1'] >= linedf.iloc[index]['x1']) & (linedf.index != index)]
            if len(hlonger_lines)>0:
                rows_to_drop.append(index)
    linedf = linedf.drop(rows_to_drop).reset_index(drop=True)
    return linedf

def standardize_lines(index, linedf):
    if index == 0:
        return 0
    if linedf.iloc[index]['width'] == 0:
    # Check for vertical lines
        if abs(math.floor(linedf.iloc[index]['x0']) - math.floor(linedf.iloc[index - 1]['x0']))<2:
            if abs(math.floor(linedf.iloc[index]['y1']) - math.floor(linedf.iloc[index - 1]['y0'])) < 2:
                return 0
            else:
                return 1
        else:
            return 1
    if linedf.iloc[index]['height'] == 0:
    # Check for horizontal lines
        if abs(math.floor(linedf.iloc[index]['y0']) - math.floor(linedf.iloc[index - 1]['y0']))< 2:
            if abs(math.floor(linedf.iloc[index]['x0']) - math.floor(linedf.iloc[index - 1]['x1']))< 2:
                return 0
            else:
                return 1
        else:
            return 1

def calculate_line_no(index, linedf):
    if index == 0:
        return 0
    else:
        line_no = sum(linedf['mark'].iloc[:index+1].tolist())
        return line_no

def merge_lines(linedf):
    linedf['line_no'] = [calculate_line_no(i, linedf) for i in range(len(linedf))]

    linedf = linedf.groupby('line_no').agg({'x0': 'min',
                                        'y0': 'min',
                                        'x1': 'max',
                                        'y1': 'max',
                                        'width': 'max',
                                        'height': 'max',
                                        'page': 'min'}).reset_index(drop=True)
    return linedf

def extract_linedf(pdf):

    pages_with_lines = list(range(get_no_pages(pdf)))
    pages_with_lines.sort()
    visible_pages_with_lines = []
    with pdfplumber.open(pdf) as pdf:
        linedf = pd.DataFrame()
        for page in pages_with_lines:
            data = pdf.pages[page]
            raw_df = pd.DataFrame(data.lines)
            if len(raw_df) < 1:
                continue
            visible_pages_with_lines.append(page)
            raw_df['page'] = page+1

            preprocessed_df = preprocessing_lines(raw_df)

            verticaldf = preprocessed_df[preprocessed_df['width']==0].sort_values(by=['x0','y0'], ascending=[True, False]).reset_index(drop=True)
            horizontaldf = preprocessed_df[preprocessed_df['height']==0].sort_values(by=['y0','x0'], ascending=[False, True]).reset_index(drop=True)

            if verticaldf.empty or horizontaldf.empty: #pages with underlines not table
                continue

            verticaldf['mark'] = verticaldf.apply(lambda row: standardize_lines(row.name, verticaldf), axis=1)
            horizontaldf['mark'] = horizontaldf.apply(lambda row: standardize_lines(row.name, horizontaldf), axis=1)

            verticaldf = merge_lines(verticaldf)
            horizontaldf = merge_lines(horizontaldf)

            df = pd.concat([verticaldf,horizontaldf], axis = 0)

            df['type'] = df['width'].apply(lambda x: "vertical" if x == 0 else "horizontal")

            linedf = pd.concat([linedf, df], ignore_index = True, axis = 0)
        
    return (linedf, visible_pages_with_lines)

def get_table_lines(page, wlinedf, pdf):

    try:
        vertical_lines = wlinedf[(wlinedf['type'] == "vertical")&(wlinedf['page']==page)]
        horizontal_lines = wlinedf[(wlinedf['type'] == "horizontal")&(wlinedf['page']==page)]
        if vertical_lines.empty or horizontal_lines.empty:
            return None, None

        page_width = get_page_info(pdf)[0]
        grant_min_y0 = float(min(vertical_lines['y0'].unique()))
        max_y1 = float(max(vertical_lines['y1'].unique()))

        max_y1_lines = vertical_lines[abs(vertical_lines['y1']-max_y1)<3]
        min_y0 = float(min(max_y1_lines['y0'].unique()))
        columns_delimiter =  ",".join([str(col) for col in vertical_lines[(vertical_lines['y1']<=max_y1+3) & (vertical_lines['y0']>=min_y0-3)]["x0"].unique().tolist()])
        columns_delimiters = [columns_delimiter]

        table_area = ",".join([str(coor) for coor in [0, max_y1, page_width, min_y0]])
        table_areas = [table_area]

        while int(min_y0) - int(grant_min_y0) > 3:
            vertical_lines = vertical_lines[vertical_lines['y1']<min_y0+3]

            max_y1 = float(max(vertical_lines['y1'].unique()))
            max_y1_lines = vertical_lines[abs(vertical_lines['y1']-max_y1)<3]
            min_y0 = float(min(max_y1_lines['y0'].unique()))

            columns_delimiter =  ",".join([str(col) for col in vertical_lines[(vertical_lines['y1']<=max_y1+3) & (vertical_lines['y0']>=min_y0-3)]["x0"].unique().tolist()])
            columns_delimiters.append(columns_delimiter)

            table_area = ",".join([str(coor) for coor in [0, max_y1, page_width, min_y0]])
            table_areas.append(table_area)
    except Exception as e:
        print(e)
        print(min_y0)
        print(grant_min_y0)
        print(tabulate(vertical_lines))
    return table_areas, columns_delimiters