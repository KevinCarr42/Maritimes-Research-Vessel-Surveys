"""
Custom functions from Notebook3_Functions
importing ipynb files is not ideal
"""
import pandas as pd

# SPECIES is faster than df, and must be imported in the function file
# easier to use as a global than to always pass dataframe (could switch to defaul?)
SPECIES = pd.read_csv('SPECIES.csv', index_col = 'index')
SPECIES.columns = ['SPEC', 'COMMON_NAME', 'SCIENTIFIC_NAME']

def get_species(species_code):
    """returns the common name of the species based on the species code"""
    return SPECIES[SPECIES.SPEC == species_code].COMMON_NAME.tolist()[0]


def search_species_by_name(name_contains):
    """returns a list of species that fit the query"""
    return SPECIES[SPECIES['COMMON_NAME'].str.contains(name_contains, case=False)]

def format_monthly(dataframe):
    return dataframe.style.format({
        'SPEC_TOTAL': '{:_.2f}'.format,
        'MONTH_TOTAL': '{:_.2f}'.format,
        'PROP_TOTAL': '{:.2%}'.format
    })


def format_yearly(dataframe):
    return dataframe.style.format({
        'SPEC_TOTAL': '{:_.2f}'.format,
        'YEAR_TOTAL': '{:_.2f}'.format,
        'PROP_TOTAL': '{:.2%}'.format
    })


def filtered_yearly(database, data_filter = None, by_weight=False):
    """
    assumes database is the main dataframe with the entire database
    data_filter needs the same dataframe name as database
        would be hard to repurpose this function, so not worth fixing
    data_filter is a boolean using the following syntax:
    data_filter = (database.DEPTH > 100) & (database.SPEED > 5) 
    """

    # total weight or total number?
    if by_weight == False:
        total_haul, total_to_drop = 'TOTNO', 'TOTWGT'
    else:
        total_haul, total_to_drop = 'TOTWGT', 'TOTNO'
    
    # columns to include
    columns = ['DATETIME', 'MISSION', 'SETNO', 'TOTWGT', 'TOTNO', 'SPEC', 'COMMON_NAME']

    # filter the data
    if type(data_filter) == type(None):
        haul_weights_filtered = database[columns].drop_duplicates()
    else:
        haul_weights_filtered = database[data_filter][columns].drop_duplicates()
      
    # weights by species
    yearly_weight_by_species = (
        haul_weights_filtered
        .drop(['SETNO', total_to_drop], axis=1)
        .groupby([haul_weights_filtered['DATETIME'].dt.year, 'SPEC']).sum()
    )
    yearly_weight_by_species = yearly_weight_by_species.reset_index(level=1)
    yearly_weight_by_species.index.names = ['YEAR']

    # total weights (filtered totals)
    yearly_weight_totals = (
        haul_weights_filtered
        .drop(['SETNO', total_to_drop, 'SPEC'], axis=1)
        .groupby([haul_weights_filtered['DATETIME'].dt.year]).sum()
    )
    yearly_weight_totals.index.names = ['YEAR']

    # join the two dataframes
    yearly_weight_by_species = (
        yearly_weight_by_species
        .merge(yearly_weight_totals, how='outer', on=['YEAR'])
        .rename(columns={f'{total_haul}_x': 'SPEC_TOTAL', f'{total_haul}_y': 'YEAR_TOTAL'})
    )

    # calculate the proportion of species to the total
    yearly_weight_by_species['PROP_TOTAL'] = (
        yearly_weight_by_species['SPEC_TOTAL'] / yearly_weight_by_species['YEAR_TOTAL']
    )

    return yearly_weight_by_species


def filtered_monthly(database, data_filter = None, by_weight=False):
    """
    assumes database is the main dataframe with the entire database
    data_filter needs the same dataframe name as database
        would be hard to repurpose this function, so not worth fixing
    data_filter is a boolean using the following syntax:
    data_filter = (database.DEPTH > 100) & (database.SPEED > 5) 
    """
    # total weight or total number?
    if by_weight == False:
        total_haul, total_to_drop = 'TOTNO', 'TOTWGT'
    else:
        total_haul, total_to_drop = 'TOTWGT', 'TOTNO'
    
    # columns to include
    columns = ['DATETIME', 'MISSION', 'SETNO', 'TOTWGT', 'TOTNO', 'SPEC', 'COMMON_NAME']

    # filter the data
    if type(data_filter) == type(None):
        haul_weights_filtered = database[columns].drop_duplicates()
    else:
        haul_weights_filtered = database[data_filter][columns].drop_duplicates()
      
    # weights by species
    monthly_weight_by_species = (
        haul_weights_filtered
        .drop(['SETNO', total_to_drop], axis=1)
        .groupby([haul_weights_filtered['DATETIME'].dt.year, haul_weights_filtered['DATETIME'].dt.month, 'SPEC']).sum()
    )
    monthly_weight_by_species = monthly_weight_by_species.reset_index(level=2)
    monthly_weight_by_species.index.names = ['YEAR', 'MONTH']

    # total weights (filtered totals)
    monthly_weight_totals = (
        haul_weights_filtered
        .drop(['SETNO', total_to_drop, 'SPEC'], axis=1)
        .groupby([haul_weights_filtered['DATETIME'].dt.year, haul_weights_filtered['DATETIME'].dt.month]).sum()
    )
    monthly_weight_totals.index.names = ['YEAR', 'MONTH']

    # join the two dataframes
    monthly_weight_by_species = monthly_weight_by_species
    monthly_weight_by_species = (
        monthly_weight_by_species
        .merge(monthly_weight_totals, how='outer', on=['YEAR', 'MONTH'])
        .rename(columns={f'{total_haul}_x': 'SPEC_TOTAL', f'{total_haul}_y': 'MONTH_TOTAL'})
    )

    # calculate the proportion of species to the total
    monthly_weight_by_species['PROP_TOTAL'] = (
        monthly_weight_by_species['SPEC_TOTAL'] / monthly_weight_by_species['MONTH_TOTAL']
    )

    return monthly_weight_by_species
    

def filter_by_species(dataframe, species_code):
    dataframe = dataframe[dataframe.SPEC == species_code].copy()
    dataframe.drop('SPEC', axis=1, inplace=True)  # remove SPEC label
    return dataframe
    
    
def graph_species(dataframe, species_code, title=None, ylabel=None):
    """
    takes in an unfiltered dataframe, filters it, and plots the data
    """
    
    # species input
    spec_no = species_code
    species = get_species(spec_no)
    
    # filter the data
    filtered_data = filter_by_species(dataframe, spec_no)
    
    # label overrides
    if title == None:
        title=f'Proportion of Haul. Species: {species}.'
    if ylabel == None:
        ylabel='Proportion of Total'
    
    # graph the proportion of haul by species by year
    dataframe[dataframe.SPEC == spec_no].plot(
        kind='bar',
        width=1,
        y='PROP_TOTAL', 
        ylabel=ylabel,
        figsize=(30, 8), 
        legend=False, 
        title=title
    );
    
    
def describe_species(dataframe, species_code):
    """descriptive stats for numeric columns"""
    
    numeric_columns = ['SPEC', 'COMMON_NAME', 'SCIENTIFIC_NAME', 'FLEN', 'FWT', 'AGE',
        'SLAT', 'SLONG', 'ELAT', 'ELONG', 'DEPTH', 'SURF_TEMP', 'BOTT_TEMP', 'BOTT_SAL']

    df_numeric = dataframe[numeric_columns][dataframe.SPEC == species_code]

    # count row is dropped for display purposes, but may be useful
    return df_numeric.describe().drop('count')


def species_counts(dataframe, species_code):
    """for object data that won't appear in df.describe() along with numeric fields"""
        
    object_columns = ['MATURITY', 'SEX', 'STRAT']
    df_object = dataframe[object_columns][dataframe.SPEC == species_code]
    
    dict_of_unique = {}
    
    for i in object_columns:
        dict_of_unique[i] = df_object[i].value_counts()
    
    return dict_of_unique


def print_species_data(dataframe, species_code):
    
    print('\n', get_species(species_code))
    
    df_describe = describe_species(dataframe, species_code)
    df_count = species_counts(dataframe, species_code)
    
    # display numeric data
    display(df_describe.drop('SPEC', axis=1).style.format({
        'FLEN': '{:_.2f}'.format,
        'FWT': '{:_.2f}'.format,
        'AGE': '{:_.2f}'.format,
        'DEPTH': '{:_.2f}'.format,
        'SURF_TEMP': '{:_.2f}'.format,
        'BOTT_TEMP': '{:_.2f}'.format,
        'BOTT_SAL': '{:_.2f}'.format,
        'SLONG': '{:_.5f}'.format,
        'SLAT': '{:_.5f}'.format
    }))
    
    # display object counts
    for key in df_count.keys():
        display(pd.DataFrame(df_count[key]))


### MAPPING FUNCTIONS TODO ###