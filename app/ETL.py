import pandas as pd
import numpy as np
import sys


def apply_median(df, columns):
    '''takes in a list of columns from a dataframe (df) and applies the median of that column where the value is a NaN'''
    for col in columns:
        df.loc[:, col] = df.loc[:, col].fillna(df.loc[:, col].median())
    return df;


def create_dummy_df(df, cat_cols, dummy_na):
    '''
    INPUT:
    df - pandas dataframe with categorical variables you want to dummy
    cat_cols - list of strings that are associated with names of the categorical columns
    dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not

    OUTPUT:
    df - a new dataframe that has the following characteristics:
            1. contains all columns that were not specified as categorical
            2. removes all the original columns in cat_cols
            3. dummy columns for each of the categorical columns in cat_cols
            4. if dummy_na is True - it also contains dummy columns for the NaN values
            5. Use a prefix of the column name with an underscore (_) for separating
    '''
    for col in cat_cols:
        try:
            # for each cat add dummy var, drop original column
            df = pd.concat([df.drop(columns=col, axis=1),
                            pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=True, dummy_na=dummy_na)],
                           axis=1)
        except:
            continue
    return df;


def convert_uint8(dataframe, col):
    '''Takes in a dataframe composed of columns with 0 to 255 integers and converts to uint8 types,
    except the list of columns passed in col'''

    dataframe_unit8 = dataframe.loc[:, dataframe.columns.difference(col)].astype('uint8')
    dataframe = dataframe.loc[:, col].merge(dataframe_unit8, left_index=True, right_index=True)
    return dataframe;


def transform_data(df):
    '''
    Input
    A portfolio type dataframe

    Output
    Pandas dataframe with no string type variables and no NaN values
    '''

    # eliminating all columns with less than 50% of the values
    less50_cols = set(df.loc[:, df.isna().mean() > .5])
    df = df.drop(columns=less50_cols)

    # list of columns to be dropped
    columns_drop = ['qt_socios_pf', 'qt_socios_pj', 'fl_matriz', 'natureza_juridica_macro',
                    'de_natureza_juridica', 'de_ramo', 'idade_emp_cat', 'dt_situacao', 'fl_st_especial',
                    'fl_email', 'fl_telefone', 'nm_segmento', 'fl_optante_simples', 'nm_micro_regiao', 'sg_uf_matriz',
                    'fl_optante_simei', 'vl_faturamento_estimado_aux', 'vl_faturamento_estimado_grupo_aux',
                    'nu_meses_rescencia', 'empsetorcensitariofaixarendapopulacao']

    # dropping the columns from the list of columns to be dropped
    df = df.drop(columns=columns_drop)

    # quick function to transform True into 1 and False into 0
    transform_boolean = lambda col: 1 if col == True else 0

    # list of columns to apply the funcion above
    boolean_cols = ['fl_me', 'fl_sa', 'fl_epp', 'fl_mei', 'fl_ltda', 'fl_rm', 'fl_spa',
                    'fl_antt', 'fl_veiculo', 'fl_simples_irregular', 'fl_passivel_iss']

    # applying the transform boolean function to the list of columns above
    for col in boolean_cols:
        df[col] = df.loc[:, col].apply(transform_boolean)

    # quick function to transform having vehicles into 1 and not having vehicles into 0
    transform_vehicles = lambda col: 0 if col == 0 else 1

    # list of vehicles columns
    vehicle_cols = ['vl_total_veiculos_leves_grupo', 'vl_total_veiculos_pesados_grupo']

    # applying the transform_vehicles function to the above list
    for col in vehicle_cols:
        df[col] = df.loc[:, col].apply(transform_vehicles)

    # list of columns to fill the NaN values with the median
    median_cols = ['idade_media_socios', 'idade_maxima_socios', 'idade_minima_socios', 'qt_socios',
                   'qt_socios_st_regular']

    # filling the NaN values with the median of the column in the list above
    apply_median(df, median_cols)

    # rounding to 2 decimal cases the idade_empresa_anos feature
    df.loc[:, 'idade_empresa_anos'] = df.loc[:, 'idade_empresa_anos'].round(decimals=2)

    # replacing the no information string with a NaN value in the de_faixa_faturamento_estimado feature
    df.loc[:, 'de_faixa_faturamento_estimado'] = df.loc[:, 'de_faixa_faturamento_estimado'].replace('SEM INFORMACAO',
                                                                                                    np.nan)

    # filling the NaN values of the column with the No information label
    df['de_saude_rescencia'] = df['de_saude_rescencia'].fillna('SEM INFORMACAO')

    # list of categorical columns to transform into dummy type columns with NaN as a feature
    dummy_cols_NA_True = ['sg_uf', 'setor', 'nm_divisao', 'de_saude_tributaria', 'de_nivel_atividade', 'nm_meso_regiao',
                          'de_faixa_faturamento_estimado', 'de_faixa_faturamento_estimado_grupo', 'de_saude_rescencia']

    # transforming the list of columns above into dummy type columns with NaN as a feature
    df = create_dummy_df(df, dummy_cols_NA_True, True)

    return df;


def extract_data(filepath):
    '''
    Input
    filepath - string of the file path where the large csv is located
    col - list of columns to be kept as non uint8 type

    Reads the data in chunks, applies the cleaning function and returns the full dataframe

    Output - pandas dataframe cleaned
    '''
    # read the large csv file with specified chunksize
    df_chunk = pd.read_csv(filepath, chunksize=10000, index_col=0)

    chunk_list = []  # append each chunk df here

    # Each chunk is in df format
    for chunk in df_chunk:
        # perform data filtering
        chunk_filter = transform_data(chunk)

        # Once the data filtering is done, append the chunk to list
        chunk_list.append(chunk_filter)

    # concat the list into dataframe
    df_concat = pd.concat(chunk_list)

    # filling the remaining NaN values with zero
    df_concat = df_concat.fillna(0)

    # list of columns that are not uint8 type
    columns = ['id', 'idade_empresa_anos']

    # dropping columns with all zero values if there are any
    df_concat = df_concat.drop(columns=df_concat.columns[(df_concat == 0).all()])

    # converting all the types to uint8, except the list of columns passed
    df_concat = convert_uint8(df_concat, columns)

    return df_concat;

def load_data(df,filepath):
    '''
    Input
    pandas dataframe and the filepath to save the dataframe
    Output
    File with the pandas dataframe
    '''
    df.to_csv(filepath, index = False)




def main():
    if len(sys.argv) == 3:
        database_filepath, save_filepath = sys.argv[1:]
        print('Loading and cleaning the data...\n    DATABASE: {}'.format(database_filepath))
        market = extract_data(database_filepath)

        print('Saving database...\n    MODEL: {}'.format(save_filepath))
        load_data(market, save_filepath)

        print('Database saved!')

    else:
        print('Please provide the filepath of the database ' \
              'as the first argument and the path to save the database ' \
              'as the second argument. \n\nExample: python ' \
              'ETL.py ../data/database.csv ../data/market_ETL.csv')


if __name__ == '__main__':
    main()