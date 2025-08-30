import psycopg2
from psycopg2 import sql, Binary
from sqlalchemy import create_engine
import os
import pandas as pd
from dotenv import load_dotenv
from tabulate import tabulate
from pathlib import Path

load_dotenv(dotenv_path=r"C:\Users\phungdoan\.vscode\project\ERS\ERS system source code\.env")

host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
dbname_ERS = os.getenv("DB_NAME_ERS")
dbname_LWS = os.getenv("DB_NAME_LWS")
dbname_PTC = os.getenv("DB_NAME_PTC")
dbname_PDF = os.getenv("DB_NAME_PDF")
username = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")

## TABLE ##
# Save tables to database and fetch data from database
def save_table_to_db(table_list, schema_name, host, port, database, username, password):
    #create connection
    conn = psycopg2.connect(
        host=host,
        port=port,
        database=database,
        user=username,
        password=password,
    )
    cur = conn.cursor()

    #create schema as a group for each version
    cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema_name)))
    conn.commit()

    #save tables into database
    engine = create_engine(f'postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}')

    for table in table_list:
        try:
            if 'Title' not in table.columns.tolist():
                table_name = table['contents'].iloc[0] + "-" + str(table.columns.tolist()[-4])+"_"+"_".join([str(c) for c in table['indicator'].unique().tolist()])+"_"+\
                    str(table[table['indicator']=='page'].reset_index(drop=True).iloc[0]['spec_value'])
            else:
                table_name = table['Title'].iloc[0]
            if len(table_name) >63:
                table_name = table_name[:60]+str("...")
            
            table['ver'] = schema_name

            if any(header=="" for header in table.columns.tolist()):
                headers = [table.columns.tolist()[:col].count("") if header =="" else header for col, header in enumerate(table.columns.tolist())]
                table.columns = headers

            table_names = get_table_names(schema_name, host=host, port=port, database=database, username=username, password=password)

            table.to_sql(table_name, engine, if_exists='replace')
            #drop table existed in db
            if table_name in table_names:
                table_existed = fetch_data_from_db(table_name, schema_name, host=host, port=port, database=database, username=username, password=password)
                if table_existed.shape[0] <= table.shape[0]:
                    Drop_table_query =f'''
                    DROP TABLE IF EXISTS "{schema_name}"."{table_name}";
                    '''
                    # Execute the query
                    cur.execute(Drop_table_query)
                    conn.commit()
                    print(f"Table {table_name} replaced")
                else:
                    continue
            #move table from public to schema
            move_table_query = f'''
            ALTER TABLE public."{table_name}" SET SCHEMA "{schema_name}";
            '''
            # Execute the query
            cur.execute(move_table_query)
            conn.commit()
        except Exception as e:
            print(e)
            print(tabulate(table, table.columns))

    print(f"Your tables is successfully imported to database {database} at schema {schema_name}")
    cur.close()
    conn.close()

# get table from db
def fetch_data_from_db(table_name, schema_name, host, port, database, username, password):
    #create connection
    conn = psycopg2.connect(
        host=host,
        port=port,
        database=database,
        user=username,
        password=password,
    )
    cur = conn.cursor()
    if len(table_name) >63:
        table_name = table_name[:60]+str("...")
    
    query = f'SELECT * FROM "{schema_name}"."{table_name}" ORDER BY "index"'
    # Execute the query
    cur.execute(query)

    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]

    # Converting the fetched data to a pandas DataFrame
    table_from_db = pd.DataFrame(rows, columns=colnames)
    cur.close()
    conn.close()

    return table_from_db

# get all schema names in a database
def get_schema_names(host, port, database, username, password):
    #create connection
    conn = psycopg2.connect(
        host=host,
        port=port,
        database=database,
        user=username,
        password=password,
    )
    cur = conn.cursor()
    query="""
    SELECT schema_name FROM information_schema.schemata;
    """
    cur.execute(query)
    schemas = cur.fetchall()
    
    schema_names = [schema[0] for schema in schemas]
    return schema_names
# get all table names in a schema
def get_table_names(schema_name, host=host, port=port, database=dbname_ERS, username=username, password=password):
    #create connection
    conn = psycopg2.connect(
        host=host,
        port=port,
        database=database,
        user=username,
        password=password,
    )
    cur = conn.cursor()
    query=f"""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = '{schema_name}';
    """
    cur.execute(query)
    tables = cur.fetchall()

    table_names = [table_name[0] for table_name in tables]
    return table_names

## IMAGE ##

def save_images_to_db(image_name, image_data, image_format, schema, host=host, port=port, database=dbname_PTC, username=username, password=password):

    conn = psycopg2.connect(host = host, port= port, database= database, user= username, password= password)
    cursor = conn.cursor()

    #create schema
    cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
    conn.commit()

    create_table_query = f"""
        CREATE TABLE IF NOT EXISTS "{schema}".pdf_images(
        id SERIAL PRIMARY KEY,
        image_name VARCHAR(255),
        image_data BYTEA,
        image_format VARCHAR(10)
        )"""

    cursor.execute(create_table_query)
    conn.commit()

    # insert data into table
    insert_query = f"""
        INSERT INTO "{schema}".pdf_images
        (image_name, image_data, image_format)"""+"""
        VALUES(%s, %s, %s)
        """

    cursor.execute(insert_query, (
        image_name,
        Binary(image_data),
        image_format
    ))
    conn.commit()
    cursor.close()
    conn.close()

def get_images_name(schema_name, host=host, port=port, database=dbname_PTC, username=username, password=password):
    conn = psycopg2.connect(
        host=host,
        port=port,
        database=database,
        user=username,
        password=password,
    )
    cursor = conn.cursor()  

    query = f'SELECT * FROM "{schema_name}"."pdf_images"'
    cursor.execute(query)

    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    images_table = pd.DataFrame(rows, columns=columns)
    
    return images_table['image_name'].tolist()

def fetch_image_data_from_db(schema_name, image_name, host=host, port=port, database=dbname_PTC, username=username, password=password):
    #create connection
    conn = psycopg2.connect(
        host=host,
        port=port,
        database=database,
        user=username,
        password=password,
    )
    cursor = conn.cursor()  
    
    query = f'SELECT * FROM "{schema_name}"."pdf_images" WHERE "image_name" = \'{image_name}\''
    # Execute the query
    cursor.execute(query)

    row = cursor.fetchall()

    image_data   = row[0][2]
    image_format = row[0][3]


    cursor.close()
    conn.close()
    return image_data, image_format

## PDF ##
# Saves raw PDF files to a  database
def save_raw_pdf_to_db(pdf_list, host=host, port=port, database=dbname_ERS, username=username, password=password):
    conn = psycopg2.connect(
        host=host,
        port=port,
        database=database,
        user=username,
        password=password,
    )
    cur = conn.cursor()

    existing_schemas = get_schema_names(host=host, port=port, database=database, username=username, password=password)
    for pdf in pdf_list:

        schema_name = pdf.split("\\")[-1].split(".")[0]
        # Create schema if it doesn't exist
        cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema_name)))
        conn.commit()

        # create table
        table_names = get_table_names(schema_name, host=host, port=port, database=database, username=username, password=password)
        if 'raw_pdf' not in table_names:
            with open(pdf, 'rb') as file:
                pdf_data = file.read()
            
            create_table_query = f"""
                CREATE TABLE IF NOT EXISTS "{schema_name}".raw_pdf (
                    index serial,
                    id SERIAL PRIMARY KEY,
                    file_name VARCHAR(255) NOT NULL,
                    pdf_data BYTEA
                );
            """
            cur.execute(create_table_query)
            conn.commit()

            insert_query = f"""
                INSERT INTO "{schema_name}".raw_pdf (file_name, pdf_data) VALUES (%s, %s);
            """
            cur.execute(insert_query, (schema_name, psycopg2.Binary(pdf_data)))
            conn.commit()
        else:
            continue

# fetches raw PDF data from the database based on schema name
def fetch_raw_pdf_from_db(schema_name, folder, host=host, port=port, database=dbname_ERS, username=username, password=password):
    conn = psycopg2.connect(host=host, port=port, database=database, user=username, password=password)
    cur = conn.cursor()
    
    schema_name = schema_name.split("\\")[-1].split(".")[0]
    query = f'SELECT "file_name", "pdf_data" FROM "{schema_name}"."raw_pdf"'
    cur.execute(query)

    row = cur.fetchall()  
    file_name, pdf_data = row[0]

    cur.close()
    conn.close()
    
    file_name = os.path.join(folder, f"{file_name}.pdf")

    with open(file_name, 'wb') as pdf_file:
        pdf_file.write(pdf_data)
        
    return file_name