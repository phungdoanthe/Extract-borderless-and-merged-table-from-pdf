from tabulate import tabulate
import pdfplumber
import pandas as pd
import re
import tempfile
import tempfile, os, shutil, io
from difflib import SequenceMatcher
from .pdf_management import *
from .database_management import *

# Assigns the most relevant content heading to a text line based on its page and vertical position
def assign_contents_for_text(row, table_of_contents):
    chapters = table_of_contents[(table_of_contents['page'] < row['page']) | \
                                                    ((table_of_contents['page'] == row['page']) & (table_of_contents['position'] >= row['top']))]
    if chapters.empty:
        return "No content found"
    chapter = chapters.iloc[-1]
    return chapter['contents']

# Extracts and filters text lines from a PDF and associates them with contents from the table of contents
def extract_text(pdf, table_of_contents):
    page_width, page_height = get_page_info(pdf)
    with pdfplumber.open(pdf) as text_pdf:
        texts = pd.DataFrame()
        for page in text_pdf.pages:
            text = page.extract_text_lines()
            lines = pd.json_normalize(text)
            lines['page'] = page.page_number
            lines['top'] = lines['top'].astype(int).apply(lambda x: int(page_height - int(x)))

            lines = lines[(lines['top']<750) & (lines['top']>75)]
            lines['bottom'] = lines['bottom'].astype(int).apply(lambda x: int(page_height - int(x)))
            texts = pd.concat([texts, lines], ignore_index=True)
        
        texts = texts[['text', 'page', 'top']].reset_index(drop=True)

        texts['contents']= texts.apply(lambda row: assign_contents_for_text(row, table_of_contents), axis = 1)
        texts = texts[texts['contents'] != "No content found"].reset_index(drop=True)
    return texts

# Compares table of contents across multiple PDFs and finds differences in headings
def compare_table_of_content(pdf_list):
    compared_content = pd.DataFrame()
    pdf_names = []
    for pdf in pdf_list:
        pdf_name = pdf.split("\\")[-1][:-4]
        pdf_names.append(pdf_name)
        table_of_contents = get_table_of_contents(pdf)

        table_of_contents['page'] = table_of_contents['page'].astype(int)
        table_of_contents['position'] = table_of_contents['position'].astype(int)
        table_of_contents['position'] = table_of_contents['position'].apply(lambda x: (int(x)//10)*10)
        table_of_contents.rename(columns={'page': f'page_{pdf_name}'}, inplace=True)
        table_of_contents.rename(columns={'position': f'position_{pdf_name}'}, inplace=True)

        if compared_content.empty:
            compared_content = table_of_contents
            continue
        
        compared_content = compared_content.merge(table_of_contents, on=['contents'], how='outer')

    missing_compared_content = compared_content.fillna('missing')

    missing_compared_content.astype(str)
    diff_content = missing_compared_content[missing_compared_content.apply(lambda x: x.str.contains("missing", case=False, na=False).any(), axis=1)].reset_index(drop=True)

    if diff_content.empty:
        return diff_content

    compared_content_on_page_position_similarity = pd.DataFrame()

    for i, pdf_name in enumerate(pdf_names):
        if i == len(pdf_names) - 1:
            break

        if i == 0:
            contents_name1 = pdf_name
            contents1_cols = ['contents'] + diff_content.filter(regex=f'{pdf_name}').columns.tolist()
            contents_df1 = diff_content[contents1_cols]
            contents_df1 = contents_df1[~contents_df1.apply(lambda row: row.astype(str).str.contains("missing", case=False, na=False).any(), axis=1)]
            contents_df1.rename(columns={'contents': f'contents_{contents_name1}'}, inplace=True)
            contents_df1 = contents_df1.dropna(how='all')
        else:
            contents_name1 = contents_name2
            contents1_cols = contents2_cols
            contents_df1 = contents_df2
        
        # get the next pdf
        contents_name2 = pdf_names[i+1]
        contents2_cols = ['contents'] + diff_content.filter(regex=f'{contents_name2}').columns.tolist()
        contents_df2 = diff_content[contents2_cols]
        contents_df2 = contents_df2[~contents_df2.apply(lambda row: row.astype(str).str.contains("missing", case=False, na=False).any(), axis=1)]
        contents_df2.rename(columns={'contents': f'contents_{contents_name2}'}, inplace=True)
        contents_df2 = contents_df2.dropna(how='all')

        # compare the contents
        compared_diff_contents = contents_df1.merge(contents_df2, how='cross')
        compared_diff_contents['position_gap'] = abs(compared_diff_contents[f'position_{contents_name1}'] - compared_diff_contents[f'position_{contents_name2}'])
        compared_diff_contents['page_gap'] = abs(compared_diff_contents[f'page_{contents_name1}'] - compared_diff_contents[f'page_{contents_name2}'])

        # filter contents based on position and page gap
        compared_diff_contents = compared_diff_contents[((compared_diff_contents['position_gap'] < 30) & (compared_diff_contents['page_gap'] == 0)) | 
                                                            ((compared_diff_contents['page_gap'] == 1) & (compared_diff_contents['position_gap'] > 500))]
        compared_diff_contents = compared_diff_contents[[f'contents_{contents_name1}', f'contents_{contents_name2}']]

        if i == 0:
            compared_content_on_page_position_similarity = compared_diff_contents
        else:
            compared_content_on_page_position_similarity = compared_content_on_page_position_similarity.merge(compared_diff_contents, how='outer')

    compared_content_on_page_position_similarity.drop_duplicates(inplace=True, keep='first')

    return compared_content_on_page_position_similarity

# Calculates similarity between two strings using SequenceMatcher
def get_simillarity(text_df1, text_df2, position_gap, page_gap):
    """
    Find similar texts in the DataFrame based on the given text.
    """
    # Normalize the input text
    text_df1 = str(text_df1).lower().strip()
    text_df2 = str(text_df2).lower().strip()
    ratio = SequenceMatcher(None, text_df1, text_df2).ratio()
    
    # Calculate the gap between the two texts
    gap_throughout_doc = abs(abs(page_gap*750)-position_gap)
    if ratio == 1:
        return ratio
                              
    if (text_df1 in text_df2) or (text_df2 in text_df1):
        if gap_throughout_doc < 50 and page_gap==0:
            ratio = 0.94

    if gap_throughout_doc > 50 and len(text_df2.split())<2 and len(text_df1.split())<2 and ratio > 0.8 and len(text_df2) != len(text_df1):
        ratio = 0

    if ratio > 0.8 and page_gap > 0 and gap_throughout_doc > 250 and ratio != 1:
        ratio = 0.76

    return ratio

# Compares multiple PDFs at the text level and organizes results for further comparison
def get_diff_text(pdf_list, folder):
    compared_texts = pd.DataFrame()
    print("fetch from db")
    pdf_list = [fetch_raw_pdf_from_db(pdf, folder) for pdf in pdf_list]

    print("compare contents")
    compared_content = compare_table_of_content(pdf_list)

    # for the compare_text
    print("compare text")
    pdf_names = []
    for i, pdf in enumerate(pdf_list):
        pdf_name = pdf.split("\\")[-1].split(".")[0]
        pdf_names.append(pdf_name)

        table_of_contents = get_table_of_contents(pdf)
        texts = extract_text(pdf, table_of_contents)
        texts = texts[['text', 'page', 'contents', 'top']]
        texts['order'] = texts.groupby(['text', 'contents']).cumcount() + 1
        texts['text_compare'] = texts['text']

        if i != 0:  
            if not compared_content.empty:
                ulti_content = dict(zip(compared_content.filter(regex=pdf_name), compared_content.iloc[:,0]))
                texts['contents_0'] = texts['contents'].apply(lambda x: ulti_content.get(x, x))
            else:
                texts['contents_0'] = texts['contents']
        else:
            texts['contents_0'] = texts['contents']

        if compared_texts.empty:
            compared_texts = texts
            compared_texts.rename(columns={'text_compare': f'text_compare_{pdf_name}'}, inplace=True)
            compared_texts.rename(columns={'contents': f'contents_{pdf_name}'}, inplace=True)
            compared_texts.rename(columns={'top': f'top_{pdf_name}'}, inplace=True)
            continue

        compared_texts = compared_texts.merge(texts, on=['text', 'page', 'contents_0','order'], how='outer')
        compared_texts.rename(columns={'text_compare': f'text_compare_{pdf_name}'}, inplace=True)
        compared_texts.rename(columns={'contents': f'contents_{pdf_name}'}, inplace=True)
        compared_texts.rename(columns={'top': f'top_{pdf_name}'}, inplace=True)

    compared_texts.fillna('missing', inplace=True)
    compared_texts = compared_texts[(compared_texts.filter(regex=r'compare').nunique(axis=1) != 1) | (compared_texts.filter(regex=r'contents').nunique(axis=1) != 1)].reset_index(drop=True)
    compared_texts=compared_texts.drop(columns=['contents_0', 'text'])
    return compared_texts, pdf_names

# Updates the content of a DataFrame based on the same content found in other PDFs
def update_same_content(diff_unique_content, pdf_names):
    for i, pdf_name in enumerate(pdf_names):
        pdf_name_content = diff_unique_content.filter(regex=pdf_name)
        same_content = pdf_name_content[pdf_name_content.apply(lambda row: row.str.contains("same", case=False, na=False).any(), axis=1)]
        if same_content.empty:
            continue
        index_row = same_content.index.tolist()
        if i == 0:
            same_pdf = pdf_names[i+1]
        else:
            same_pdf = pdf_names[i-1]

        update_content = diff_unique_content.filter(regex=same_pdf).loc[index_row]
        update_content_cols = [col.replace(same_pdf, pdf_name) for col in update_content.columns]
        update_content.columns = update_content_cols

        diff_unique_content.update(update_content)
    return diff_unique_content

# updates the text differences between two PDFs based on their similarity and gap
def get_diff_text_by_similarity(diff_texts, pdf_names):
    
    diff_unique_content = pd.DataFrame()
    for i in range(1, len(pdf_names)):
        
        # get the first pdf
        if i == 1:
            pdf_name1 = pdf_names[i-1]
            pdf1_cols = [col for col in diff_texts.columns.tolist() if pdf_name1 in col] + ['page', 'order']
            pdf_df1 = diff_texts[pdf1_cols]

            pdf_df1 = pdf_df1[~pdf_df1.apply(lambda row: row.str.contains("missing", case=False, na=False).any(), axis=1)]
            
            str_cols1 = [col for col in pdf1_cols if pdf_name1 in col and 'top' not in col]
            pdf_df1[str_cols1] = pdf_df1[str_cols1].astype(str)

            pdf_df1[f'top_{pdf_name1}'] = pdf_df1[f'top_{pdf_name1}'].astype(int)
            pdf_df1['page'] = pdf_df1['page'].astype(int)
            pdf_df1['order'] = pdf_df1['order'].astype(int)

            pdf_df1.rename(columns={'page' : f'page_{pdf_name1}'}, inplace=True)
            pdf_df1.rename(columns={'order' : f'order_{pdf_name1}'}, inplace=True)

            pdf_df1[f'text_compare_{pdf_name1}'] = pdf_df1[f'text_compare_{pdf_name1}'].apply(lambda x: refine_table_name((re.sub(r'[^a-zA-Z0-9\s]{3,}', "", x))).strip())

            pdf_df1 = pdf_df1.dropna(how='all')

        else:
            pdf_name1 = pdf_name2
            pdf1_cols = pdf2_cols
            pdf_df1 = pdf_df2
        
        # get the next pdf
        pdf_name2 = pdf_names[i]
        pdf2_cols = [col for col in diff_texts.columns.tolist() if pdf_name2 in col] + ['page', 'order']
        pdf_df2 = diff_texts[pdf2_cols]

        pdf_df2 = pdf_df2[~pdf_df2.apply(lambda row: row.str.contains("missing", case=False, na=False).any(), axis=1)]
        str_cols2 = [col for col in pdf1_cols if pdf_name2 in col and 'top']
        pdf_df1[str_cols2] = pdf_df2[str_cols2].astype(str)

        pdf_df2[f'top_{pdf_name2}'] = pdf_df2[f'top_{pdf_name2}'].astype(int)
        pdf_df2['page'] = pdf_df2['page'].astype(int)
        pdf_df2['order'] = pdf_df2['order'].astype(int)

        pdf_df2.rename(columns={'page' : f'page_{pdf_name2}'}, inplace=True)
        pdf_df2.rename(columns={'order' : f'order_{pdf_name2}'}, inplace=True)

        pdf_df2[f'text_compare_{pdf_name2}'] = pdf_df2[f'text_compare_{pdf_name2}'].apply(lambda x: refine_table_name((re.sub(r'[^a-zA-Z0-9\s]{3,}', "", x))).strip())

        pdf_df2 = pdf_df2.dropna(how='all')

        # join the two pdfs
        pdf_df1_0 = pdf_df1.copy()
        pdf_df1_0['page_0'] = pdf_df1_0[f'page_{pdf_name1}']

        pdf_df2_m1 = pdf_df2.copy()
        pdf_df2_m1['page_0'] = pdf_df2_m1[f'page_{pdf_name2}'].apply(lambda x: x-1 if x>1 else x)
        pdf_df2_a1 = pdf_df2.copy()
        pdf_df2_a1['page_0'] = pdf_df2_a1[f'page_{pdf_name2}'].apply(lambda x: x+1)
        pdf_df2_0 = pdf_df2.copy()
        pdf_df2_0['page_0'] = pdf_df2_0[f'page_{pdf_name2}']
        pdf_df2_0 = pd.concat([pdf_df2_0, pdf_df2_m1, pdf_df2_a1], axis=0, ignore_index=True)

        pdf1_cols = pdf_df1_0.columns.tolist()
        pdf2_cols = pdf_df2_0.columns.tolist()

        join_i = pdf_df1_0.merge(pdf_df2_0, on=['page_0'], how='outer')

        # calculate gap between pages
        join_i['page_gap'] = join_i.apply(lambda row: abs(float(row[f'page_{pdf_name1}']) - float(row[f'page_{pdf_name2}'])), axis=1)

        join_i = join_i[(join_i['page_gap']<2)]
        if join_i.empty:
            continue
        # calculate gap between texts
        join_i['position_gap'] = join_i.apply(lambda row: abs(float(row[f'top_{pdf_name1}']) - float(row[f'top_{pdf_name2}'])), axis=1)

        # calculate similarity
        join_i['similarity'] = join_i.apply(lambda row: get_simillarity(row[f'text_compare_{pdf_name1}'], row[f'text_compare_{pdf_name2}'], row['position_gap'], row['page_gap']), axis=1)
        
        # print(tabulate(join_i, join_i.columns, "grid"))
        # filter the join_i dataframe
        join_i = join_i[(join_i['similarity'] > 0.75)].reset_index()
        join_i_top_pdf1 = join_i.groupby(pdf1_cols).agg({'similarity': 'max'}).reset_index()
        join_i_top_pdf2 = join_i.groupby(pdf2_cols).agg({'similarity': 'max'}).reset_index()
        # print(tabulate(join_i, join_i.columns, "grid"))
        # merge the join_i dataframe with the top values
        cols_to_join_pdf1 = pdf1_cols + ['similarity']
        cols_to_join_pdf2 = pdf2_cols + ['similarity']

        join_i = join_i.merge(join_i_top_pdf1, on=cols_to_join_pdf1, how='inner')
        join_i = join_i.merge(join_i_top_pdf2, on=cols_to_join_pdf2, how='inner')
        # print(tabulate(join_i, join_i.columns, "grid"))
        join_i = join_i.sort_values(by='index').reset_index(drop=True).drop(columns=['index', 'similarity'])

        join_i = join_i[(join_i[f'order_{pdf_name1}'] == join_i[f'order_{pdf_name2}']) | 
                        ((join_i[f'top_{pdf_name1}'] == join_i[f'top_{pdf_name2}']) & (join_i[f'page_{pdf_name1}'] == join_i[f'page_{pdf_name2}']))]
        # print(tabulate(join_i, join_i.columns, "grid"))
        # get the new or removed text lines
        for pdf_df in [pdf_df1, pdf_df2]:
            # remove duplicates
            join_i_pdf = join_i[pdf_df.columns]
            join_i_pdf = pd.concat([join_i_pdf, pdf_df], axis = 0, ignore_index = True)
            join_i_pdf['count'] = join_i_pdf.groupby(join_i_pdf.columns.tolist()).transform('size')
            join_i_pdf = join_i_pdf[join_i_pdf['count']<2]
            join_i = pd.concat([join_i, join_i_pdf], axis = 0, ignore_index = True).reset_index(drop=True)
            sort_cols =  ['page_0'] + join_i.filter(regex='top').columns.tolist()
            join_i = join_i.sort_values(by=sort_cols, ascending=False).reset_index(drop=True)
            join_i = join_i.fillna('missing')

        # print(tabulate(join_i, join_i.columns, "grid"))
        # filter out rows where all text_compare columns are the same
        join_i = join_i[join_i.filter(regex='text_compare').nunique(axis=1) != 1]
        join_i = join_i.astype(str)

        # merge the join_i dataframe with the diff_unique_content dataframe
        if diff_unique_content.empty:
            diff_unique_content = join_i
        else:
            # fill out missing so it will not match with missing of previous result
            filtered_join_i = join_i[~join_i.apply(lambda row: row.str.contains("missing", case=False, na=False).any(), axis=1)]
            diff_unique_content = diff_unique_content.merge(filtered_join_i, on=pdf1_cols, how='outer')
            diff_unique_content = pd.concat([diff_unique_content, join_i] , axis = 0, ignore_index=True)
            diff_unique_content = diff_unique_content.drop_duplicates(subset=join_i.columns.tolist(), keep='first').reset_index(drop=True)

    return diff_unique_content

# Main function to compare texts across multiple PDFs and identify similarities and differences
def compare_texts(raw_pdf_list, host=host, port=port, database=dbname_ERS, username=username, password=password):
    
    # create a temporary folder to store the pdfs
    temp_pdf_folder = tempfile.mkdtemp()

    # save raw pdf
    pdf_list = []
    for pdf in raw_pdf_list:
        pdf_name = pdf.split("\\")[-1].split(".")[0]
        table_names = get_table_names(pdf_name, host=host, port=port, database=database, username=username, password=password)
        if 'raw_pdf' not in table_names:
            save_raw_pdf_to_db(raw_pdf_list)

        pdf_data = fetch_raw_pdf_from_db(pdf_name, temp_pdf_folder, host=host, port=port, database=database, username=username, password=password)
        pdf_list.append(pdf_data)

    print("Getting diff texts done")
    # get diff result before transform
    diff_texts, pdf_names = get_diff_text(pdf_list, temp_pdf_folder)

    #compare texts that change but can not on the same row
    diff_unique_content = get_diff_text_by_similarity(diff_texts, pdf_names)
    print("Getting diff texts by similarity done")
    # print(tabulate(diff_unique_content, headers='keys', tablefmt='psql'))
    # drop unnecessary columns
    print("Drop unnecessary columns")
    columns_to_drop = diff_unique_content.filter(regex=r"gap").columns.tolist() + [col for col in diff_unique_content.columns if any(metric in col for metric in ['top', 'order', 'count'])]
    diff_unique_content = diff_unique_content.drop(columns=columns_to_drop, errors='ignore')

    # fill missing values and drop unnecessary columns
    diff_unique_content = diff_unique_content.fillna("same")
    print("Fill missing values done")
    # print(tabulate(diff_unique_content, headers='keys', tablefmt='psql'))
    # update the same content
    diff_unique_content = update_same_content(diff_unique_content, pdf_names)
    diff_unique_content.drop(columns=['page_0'], inplace=True, errors='ignore')
    diff_unique_content_cols = diff_unique_content.columns.tolist().sort()
    diff_unique_content = diff_unique_content

    print("Update same content done")
    shutil.rmtree(temp_pdf_folder, ignore_errors=True)
    
    return diff_unique_content