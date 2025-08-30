from .compare_texts import *
from .extract_normalize_tables import *
from english_words import get_english_words_set
from datetime import datetime
from tabulate import tabulate
import nltk
import pandas as pd
import psycopg2
from psycopg2 import sql
from utiliti_functions.pdf_management import *
from utiliti_functions.database_management import *

#extracting and standardizing
def extract_tables(extracted_pdf):
    pdf = extracted_pdf
    pdfname = pdf.split("\\")[-1][:-4]
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    print("extracting table of contents...")
    table_of_contents = get_table_of_contents(pdf)
    print("extracting list of lines...")
    wlinedf, pages_with_lines = extract_linedf(pdf)
    print("extracting tables name...")
    tables_name = get_all_table_name(pdf)
    print(f"extracting raw tables from {pdfname}...")
    if wlinedf.empty:
        transformed_table_list =  extract_table_RDD(pdf)
    else:
        table_list = extract_raw_tables(pdf, wlinedf)#, start_page = start_page, end_page = 0)
        print("normalizing tables..")
        normalized_tables = normalize_tables(table_list, wlinedf, pages_with_lines, table_of_contents, tables_name)     
        print("transforming tables..")
        transformed_table_list = transform_tables(normalized_tables)
        print("finish extracting tables!")

    return transformed_table_list

#get simmilar string
def similar_string(str, list_strs):
    top_matches = process.extract(str, list_strs, scorer=fuzz.token_sort_ratio, limit=5)
    print(str)
    print(top_matches)
    best_match = [match for match, score, _ in top_matches if score > 80]
    if best_match:
        return best_match[0], True
    else:
        return str, False

# compare tables
def compare_tables(tables):
    table_list = []
    for table in tables:
        table = table.map(lambda x: x.replace("\n", " ") if isinstance(x, str) else x)
        table = table.map(lambda x: x.replace("  ", " ") if isinstance(x, str) else x)
        table_list.append(table)
    
    merged_table = pd.DataFrame()
    try:
        table_0 = table_list[0].drop(columns=['ver', 'index'], errors='ignore')
        indicators_0 = table_0['indicator'].tolist()
    except KeyError as e:
        return None
        
    for i, table in enumerate(table_list):
        ver_i = "_ver" + (table.iloc[0]['ver'].split("ver")[-1])

        table_i = table.drop(columns=['ver', 'index'])
        table_i['order'] = table_i.groupby(table_i.columns.tolist()[:-1]).cumcount()+1
        try:
            if i == 0 :
                merged_table = table_i
                merged_table = merged_table.rename(columns={"spec_value":f"spec_value{ver_i}"})
                continue
            else:
                indicators_i = table_i['indicator'].tolist()
                if indicators_0 != indicators_i and len(indicators_0) == len(indicators_i):
                    table_i['indicator'] = indicators_0

                key_i = table_i.columns.tolist()[:-2]+['order']
                merged_table = merged_table.merge(table_i, on=key_i, how='outer')
                merged_table = merged_table.rename(columns={"spec_value":f"spec_value{ver_i}"})
        except KeyError as e:
            print(e)
            print(tabulate(table_i.columns))
            print(tabulate(merged_table.columns))

    merged_table = merged_table.fillna('missing')
    merged_table = merged_table.replace(r"^-$", "", regex=True)
    spec_cols = merged_table.filter(regex=r'spec_value').columns
    compared_table = merged_table[merged_table[spec_cols].nunique(axis=1)!=1]

    compared_table = compared_table.drop(columns=['order'])

    parameter_cols = merged_table.filter(regex=r'P\s*?a\s*?r\s*?a\s*?m\s*?e\s*?t\s*?e\s*?r_').columns
    if len(parameter_cols)>=2:
        compared_table = compared_table.drop(columns = parameter_cols)

    return compared_table

# compare table cells by similarity
def compare_table_cells_by_similarity(diff_cells, pdf_names):
    
    for col in diff_cells.filter(regex='spec_value').columns.tolist():
        if diff_cells[col].astype(str).str.contains("missing", case=False, na=False).any():
            common_cols = [col for col in diff_cells.columns.tolist() if "spec_value" not in col]
        else:
            return diff_cells
    
    diff_unique_content = pd.DataFrame()
    for i in range(1, len(pdf_names)):
        
        # get the first pdf
        if i == 1:
            pdf_name1 = pdf_names[i-1]
            pdf1_cols = common_cols + diff_cells.filter(regex=pdf_name1).columns.tolist()
            pdf_df1 = diff_cells[pdf1_cols]

            pdf_df1 = pdf_df1[~pdf_df1.apply(lambda row: row.str.contains("missing", case=False, na=False).any(), axis=1)]
            pdf_df1 = pdf_df1.dropna(how='all')
            pdf_df1['indicator'] = pdf_df1['indicator'].apply(lambda x: x.replace(" ", "").lower())
        else:
            pdf1_cols = pdf2_cols
            pdf_df1 = pdf_df2
        
        # get the next pdf
        pdf_name2 = pdf_names[i]

        pdf2_cols = common_cols + diff_cells.filter(regex=pdf_name2).columns.tolist()
        pdf_df2 = diff_cells[pdf2_cols]

        pdf_df2 = pdf_df2[~pdf_df2.apply(lambda row: row.str.contains("missing", case=False, na=False).any(), axis=1)]
        pdf_df2 = pdf_df2.dropna(how='all')
        pdf_df2['indicator'] = pdf_df2['indicator'].apply(lambda x: x.replace(" ", "").lower())
        
        join_cols = common_cols

        if pdf_df1['contents'].iloc[0] != pdf_df2['contents'].iloc[0]:
            join_cols = [col for col in join_cols if col != 'contents']
            pdf_df1.rename(columns={'contents': f'contents_{pdf_name1}'}, inplace=True)
            pdf_df2.rename(columns={'contents': f'contents_{pdf_name2}'}, inplace=True)

        if pdf_df1['Title'].iloc[0] != pdf_df2['Title'].iloc[0]:
            join_cols = [col for col in join_cols if col != 'Title']
            pdf_df1.rename(columns={'Title': f'Title{pdf_name1}'}, inplace=True)
            pdf_df2.rename(columns={'Title': f'Title{pdf_name2}'}, inplace=True)

        # join the two pdfs
        join_i = pdf_df1.merge(pdf_df2, on=join_cols, how='outer')
        
        # calculate similarity and gap
        col_cell_compare_pdf1 = pdf_df1.filter(regex='spec_value').columns.tolist()[0]
        col_cell_compare_pdf2 = pdf_df2.filter(regex='spec_value').columns.tolist()[0]
        join_i['similarity'] = join_i.apply(lambda row: get_simillarity(row[col_cell_compare_pdf1], row[col_cell_compare_pdf2], 0, 0), axis=1)
        # print(tabulate(join_i, headers='keys', tablefmt='psql'))

        # filter the join_i dataframe
        join_i = join_i[join_i['similarity'] > 0.75].reset_index()
        join_i_top = join_i.groupby(join_cols).agg({'similarity': 'max'}).reset_index()
        # print(tabulate(join_i, headers='keys', tablefmt='psql'))

        # merge the join_i dataframe with the top values
        cols_to_join = join_cols + ['similarity']
        join_i = join_i.merge(join_i_top, on=cols_to_join, how='inner')
        join_i = join_i.sort_values(by='index').reset_index(drop=True).drop(columns=['index', 'similarity'])
        # print(tabulate(join_i, headers='keys', tablefmt='psql'))

        # get the new or removed text lines
        for pdf_df in [pdf_df1, pdf_df2]:
            if pdf_df.shape[0] != join_i.shape[0]:
                join_i = pd.concat([join_i, pdf_df], axis = 0, ignore_index=True)
                # remove duplicates
                join_i = join_i.drop_duplicates(subset=pdf_df.columns.tolist()[1:], keep='first').reset_index(drop=True)
                join_i = join_i.fillna('missing')

        # filter out rows where all text_compare columns are the same
        join_i = join_i[join_i.filter(regex='spec_value').nunique(axis=1) != 1]
        join_i = join_i.astype(str)

        # merge the join_i dataframe with the diff_unique_content dataframe
        if diff_unique_content.empty:
            diff_unique_content = join_i
        else:
            diff_unique_content = diff_unique_content.merge(join_i, on=pdf1_cols[1:], how='outer')
    return diff_unique_content

# compare pdf
def compare_pdf(pdf_list, host=host, port=port, database=dbname_ERS, username=username, password=password):
    pd.options.mode.chained_assignment = None
    if len(pdf_list)<1:
        print("Need 2 or more pdfs")
    schema_names_in_db = get_schema_names(host=host, port=port, database=database, username=username, password=password)
    database_name = []
    schema_names = []
    for pdf in pdf_list:
        schema_name = pdf.split("\\")[-1][:-4]
        #contains schemas name
        schema_names.append(schema_name)
        if not schema_name in schema_names_in_db:
            table_list = extract_tables(pdf)
            #save data to database
            save_table_to_db(table_list, schema_name, host=host, port=port, database=database, username=username, password=password)
        #contains tables name
        schema = get_table_names(schema_name, host=host, port=port, database=database, username=username, password=password)
        schema = [table for table in schema if table not in database_name]
        #list in list
        database_name.extend(schema)

    #compare
    database_name.sort(key=len, reverse=True)
    longest_schema = database_name[0]
    compared_table_list = []

    #extract table from db based on table name
    exist_tables = []
    notifications = []

    for table_name in database_name:
        if table_name in exist_tables:
            continue

        table_list = []

        for schema_name in schema_names:            
            try:
                table = fetch_data_from_db(table_name, schema_name, host=host, port=port, database=database, username=username, password=password)
                table_list.append(table)
                exist_tables.append(table_name)

            except psycopg2.errors.UndefinedTable:
                try:
                    db_table_list = get_table_names(schema_name, host=host, port=port, database=database, username=username, password=password)
                    similar_table_name, flag = similar_string(table_name, db_table_list)

                    table = fetch_data_from_db(similar_table_name, schema_name, host=host, port=port, database=database, username=username, password=password)
                    table_list.append(table)
                    exist_tables.append(similar_table_name)

                except psycopg2.errors.UndefinedTable as e:

                    notifications.append(f"{schema_name} does not have table {table_name}")

        #compare table
        if len(table_list)< 2:
            continue
        else:
            mix_compared_table = compare_tables(table_list)
            if mix_compared_table is None:
                notifications.append(f"Table {table_name} cannot be compared")
            elif mix_compared_table.empty:
                continue
            else:
                compared_table = compare_table_cells_by_similarity(mix_compared_table, schema_names)
                compared_table_list.append(compared_table)
        
        # notify about non exist tables
        non_exist_tables = []
        for list in database_name:
            non_exist_tables.extend([table_name for table_name in list if table_name not in exist_tables])

    return compared_table_list, notifications
