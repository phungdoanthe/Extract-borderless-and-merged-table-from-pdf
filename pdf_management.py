import ftfy
import pdfplumber
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from english_words import get_english_words_set
import tempfile, os, shutil, io, re
from camelot import utils

def refine_words(table):
    """Vectorized version of refine_words"""
    return table.map(lambda x: ftfy.fix_text(x) if isinstance(x, str) else x)

def refine_table_name(table_name):
    '''
    some table names are duplicated in like 'TTaabbllee:: ....'
    '''
    checked_table_name = table_name.replace(" ", "  ") #because " " is not duplicated
    #check for duplicated table name
    index = [i for i in range(0,len(checked_table_name),2)]
    a = "".join([c for i, c in enumerate(checked_table_name) if i in index])
    b = "".join([c for i, c in enumerate(checked_table_name) if i not in index])
    if a==b:
        return a
    else:
        return table_name

def get_page_info(pdf):
    '''
    get the length of the pdf page to adjust coordinates  because we use 2 different libs for extracting raw table and other information
    '''
    layout, dim = utils.get_page_layout(pdf)
    page_width, page_height = int(dim[0]), int(dim[1])
    return page_width, page_height

def get_no_pages(pdf)-> int:
        with pdfplumber.open(pdf) as pdf:
                return len(pdf.pages)

def get_table_of_contents(pdf):
    '''
    get table of contents. This could be passed in normalize table function if not required
    because each doc will have different table of contents so is it crucial to identify where table of contents is
    '''
    #get start and end page of table of contents
    with pdfplumber.open(pdf) as content_pdf:
        texts = pd.DataFrame()
        for page in content_pdf.pages:
            text = page.extract_text_lines()
            lines = pd.json_normalize(text)
            lines = lines[(lines['top']<720)&(lines['top']>40)]
            lines['page'] = page.page_number
            texts = pd.concat([texts, lines], ignore_index=True)
        texts['font_size'] = texts['chars'].apply(lambda x:x[0]['y1'] - x[0]['y0'])
        texts = texts[['text', 'page', 'font_size']]
        texts = refine_words(texts)
        contents_start_info = texts[texts['text'].str.contains(r"(?i)contents", regex=True)].iloc[0]
        font_size = contents_start_info['font_size']
        start_page = contents_start_info['page']-1
        end_page = texts[(texts['font_size'] == font_size) & (texts.index > contents_start_info.name)]['page'].min()-1

    #get table of contents page
    page_len = get_page_info(pdf)[1]
    with pdfplumber.open(pdf) as pdf_table_of_contents:
        table_of_contents = pd.DataFrame()
        for i in range(start_page,end_page):
            table_of_content = pd.json_normalize(pdf_table_of_contents.pages[i].extract_text_lines())
            table_of_contents = pd.concat([table_of_contents, table_of_content], ignore_index=True)

    #only take lines that like 'contents....page' or 'appedix  page'
    table_of_contents = table_of_contents[(table_of_contents['text'].str.contains("\.\.\.")) | (table_of_contents['text'].str.contains(r"^Appendix", case=True, regex=True))]

    #standardize table of contents
    table_of_contents['text'] = table_of_contents['text'].str.replace(r"(\d) (\d)", r"\1\2", regex=True)
    table_of_contents.iloc[:, 0] = table_of_contents.iloc[:, 0].apply(lambda x: str(x).replace("\n", ".").replace("\t","."))
    table_of_contents.rename(columns={'text': 'contents'}, inplace = True)
    #remove "." and pick out page and contents
    table_of_contents['page'] = table_of_contents.iloc[:, 0].apply(lambda x: int(re.split(r"\.{2,}", x)[-1]) if "Appendix" not in x else int(x.split(" ")[-1]))
    table_of_contents['contents'] = table_of_contents.iloc[:, 0].apply(lambda x: re.split(r"\.{2,}", str(x))[0].strip("\n ") if "Appendix" not in x else re.sub(r"\d", "", str(x)).strip())
    # print(tabulate(table_of_contents, table_of_contents.columns, "gird"))
    #refine ligatures
    table_of_contents = table_of_contents[['contents','page']]
    table_of_contents = refine_words(table_of_contents)

    #get positon of each content
    contents = [text.lower() for text in table_of_contents['contents'].tolist()]
    pages = list(set(table_of_contents['page'].tolist()))
    pages.sort()
    #extract all the text in the pdf to get the coordinate of words
    with pdfplumber.open(pdf) as pdf:
        raw_contents = pd.DataFrame()
        for page in pages:
            content = pd.json_normalize(pdf.pages[page-1].extract_text_lines())
            content['page'] = page
            raw_contents = pd.concat([raw_contents, content], ignore_index= True)
    raw_contents = raw_contents[['text','top','page']]

    #only take lines that are title of the section
    new_table_of_contents = raw_contents[(raw_contents['text'].str.lower().isin(contents)) | (raw_contents['text'].str.contains(r"^Appendix\s\w", regex=True))].reset_index(drop=True)

    #adjust the position because it is opposite between pdfplumber and camlot when it comes to coordinates
    new_table_of_contents['top'] = new_table_of_contents['top'].apply(lambda x: page_len-int(x))
    new_table_of_contents['page'] = new_table_of_contents['page'].astype(int)
    new_table_of_contents.rename(columns={'text': 'contents', 'top': 'position'}, inplace = True)
    #remove line that contains the word 'table' but not table name
    lines_to_drop = new_table_of_contents[new_table_of_contents['contents'].str.contains("table", case=False)].index
    new_table_of_contents = new_table_of_contents.drop(lines_to_drop).reset_index(drop=True)
    # print(tabulate(new_table_of_contents, new_table_of_contents.columns, "gird"))
    return new_table_of_contents

def normalize_sentence(sentence):

    word_set = get_english_words_set(['web2'], lower=True)

    if "E-" in sentence:
        return sentence

    if "_" in sentence:
        return sentence.replace(" ", "")

    sentence = sentence.replace("\n", "\n ")
    sentence = sentence.split(" ")

    new_sentence_list = []
    lemmatizer = WordNetLemmatizer()

    for i in range(len(sentence)):
        word = sentence[i]
        if i ==0:
            new_sentence_list.append(word)
            continue
        word0 = new_sentence_list[-1]
        if not re.sub(r'[^a-zA-Z\s]', '', word0).strip("\n ") or not re.sub(r'[^a-zA-Z\s]', '', word).strip("\n "):
            new_sentence_list.append(word)
            continue
        corrected_word = re.sub(r'[^a-zA-Z\s]', '', word0).strip("\n ").lower()+re.sub(r'[^a-zA-Z\s]', '', word).strip("\n ").lower()
        if corrected_word in word_set:
            new_sentence_list[-1] = word0.strip("\n ")+word.strip("\n ")
        elif any(word == corrected_word[-len(word):] for word in ['ing', 'es', 'ed', 's', 'ize', 'ify', 'en']):
            restored_word = lemmatizer.lemmatize(corrected_word, pos=wordnet.VERB)
            if restored_word in word_set:
                new_sentence_list[-1] = word0.strip("\n ")+word.strip("\n ")
            elif corrected_word[-1] =="s":
                if corrected_word[:-1] in word_set:
                    new_sentence_list[-1] = word0.strip("\n ")+word.strip("\n ")
                else:
                    new_sentence_list.append(word)
            else:
                new_sentence_list.append(word)
        else:
            new_sentence_list.append(word)
    new_sentence = " ".join(new_sentence_list)
    new_sentence = new_sentence.replace("\n ","\n")
    return new_sentence

def get_all_table_name(pdf):
    page_len = get_page_info(pdf)[1]
    with pdfplumber.open(pdf, ) as pdf:
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
            tables_name = lines.loc[(lines['text'].str.contains(r'^Table ', regex=True) |\
                                    lines['text'].str.contains(r'^TTaabbllee', regex=True)) |\
                                    lines['text'].str.contains(r'^Relay Lens Set-Up Requirements', regex=True)]\
                                            [['text', 'page', 'top']]
            tables_name['text'] = tables_name['text'].apply(lambda x: x.split(" Rev ")[0] if " Rev " in x else x)
            tables_name['top'] = tables_name['top'].astype(int).apply(lambda x: page_len-int(x))
            tables_name.rename(columns={'text': 'table name'}, inplace=True)
            tables_name['table name'] = tables_name['table name'].apply(lambda x: refine_table_name(x))
            return tables_name