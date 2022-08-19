"""
Custom functions from Notebook3_Functions
Also includes imported dataframes from CSV tables and joined database
"""

###### IMPORT MODULES ######

import pandas as pd
import plotly.express as px


###### IMPORT DATA ######

# entire database in one dataframe (df for simplicity)
# NOTE this file IS NOT ON GITHUB, it is too big, so it is in gitignore
# run the RV Database notebook to create this file from the cleaned tables
df = pd.read_csv('RV_DATABASE.csv', 
    dtype={'MATURITY':object, 
           'STRAT':object, 
           'TOTNO':'Int64', 
           'SPEC':'Int64',
           'SPECIMEN_ID':'Int64'},
    parse_dates=['DATETIME'])

# sort by datetime and reindex
df = df.sort_values('DATETIME').reset_index(drop=True)

# individual tables used to define the database
SPECIES = pd.read_csv('SPECIES.csv', index_col = 'index')
SPECIES.columns = ['SPEC', 'COMMON_NAME', 'SCIENTIFIC_NAME']
MISSIONS = pd.read_csv('MISSIONS.csv')
GSCAT = pd.read_csv('GSCAT.csv')
GSINF = pd.read_csv('GSINF.csv', index_col='date and time', parse_dates=['date and time'])  # parse index as pd.datetime format
GSINF.index.name = 'DATETIME'
GSDET = pd.read_csv('GSDET.csv')


###### FUNCTIONS ######

def get_species(species_code, SPECIES=SPECIES):
    """returns the common name of the species based on the species code"""
    return SPECIES[SPECIES.SPEC == species_code].COMMON_NAME.tolist()[0]


def species_codes_by_percentile(dataframe, percentile_as_decimal, ascending=False):
    """
    input 0.1 to get top 10% of species, 1 returns all species
    based on specimen counts
    ascending=True will return the bottom percentile_as_decimal of species
    """
    
    spec_counts = pd.DataFrame(dataframe.SPEC.value_counts(ascending=ascending))
    spec_list = list(spec_counts.SPEC.index)
    number_of_species = len(spec_list)
    return spec_list[0:int(number_of_species*percentile_as_decimal)]


def search_species_by_name(name_contains, SPECIES=SPECIES):
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


def top_species_by_attribute(dataframe, attribute='DEPTH', aggregation='mean', how_many=10, min_species=None, date_min=None, date_max=None):
    """
    top_species_by_attribute(dataframe, attribute='DEPTH', aggregation='mean', how_many=10, min_species=None, date_min=None, date_max=None)
        attribute = 'DEPTH'  # default
            attribute = None  # for a simple count of species
            this can be done more simply with the top_x_species() function
        aggregation = 'mean'
        how_many = 10  # top how_many species
        min_species = 1000  # min specimens per species, ie, ignore rare species less than min_species in number
        filter by dates in string format (formats recognisable by pandas to_datetime() function)
    """
    
    # import dataframe and filter
    # NOTE: if statements speed up the operation ~400ms each on my computer
    
    # filter by min_species
    if (min_species == 0) or (min_species is None):
        top_species = dataframe
    else:
        top_species = filter_by_min_species(dataframe, min_species=min_species)  
    
    # filter by dates
    if (date_min == None) and (date_max == None):
        pass
    else:
        top_species = filter_dates(top_species, date_min=date_min, date_max=date_max)  # by dates
    
    # filter by species and attribute
    if attribute == None:
        top_species = pd.DataFrame(dataframe.SPEC.value_counts().head(how_many)).rename(columns={'SPEC':'COUNT'})
        top_species['SPEC'] = top_species.index
        top_species['NAME'] = top_species['SPEC'].apply(get_species)
        top_species = top_species[['SPEC', 'NAME', 'COUNT']].set_index('SPEC')
    else:
        top_species = pd.DataFrame(top_species.groupby('SPEC')[attribute].agg(aggregation).sort_values(ascending=False).head(how_many))
        top_species['CODE'] = top_species.index
        top_species['NAME'] = top_species['CODE'].apply(get_species)
        top_species = top_species[['NAME', attribute]].rename(columns={attribute: f'{aggregation}_{attribute}'.upper()})
        top_species['COUNTS'] = dataframe.SPEC.value_counts()
    
    return top_species


def top_x_species(dataframe, how_many=10):
    """top 10 most common species by count"""
    return top_species_by_attribute(dataframe, how_many=how_many, attribute=None, min_species=None)


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


def average_geo(dataframe):
    """
    averages SLAT and ELAT & SLONG and ELONG
    null values do not get averaged
    """
    geo_df = dataframe
    geo_df['LAT'] = geo_df[['SLAT', 'ELAT']].mean(axis=1)
    geo_df['LONG'] = geo_df[['SLONG', 'ELONG']].mean(axis=1)
    return geo_df.drop(['SLAT', 'ELAT', 'SLONG', 'ELONG'], axis=1)


def filter_dates(dataframe, date_min=None, date_max=None):
    """
    date_min and date_max are strings, formatted to be interpreted by pandas to_datetime() function
    """
    # initialise dates
    datetime_min, datetime_max = dataframe.DATETIME.min(), dataframe.DATETIME.max()
    
    # if dates are inputted, create filters for dataframe
    if date_min:
        datetime_min = pd.to_datetime(date_min)
    if date_max:
        datetime_max = pd.to_datetime(date_max)
        
    # filter dataframe by dates
    return dataframe[(dataframe.DATETIME >= datetime_min) & (dataframe.DATETIME <= datetime_max)]


def scatterplot_species(dataframe, species_code, x='DATETIME', y='DEPTH', date_min=None, date_max=None):
    """scatterplot of 2 species attributes: defaults to plotting depth vs time"""
    plot_df = filter_dates(dataframe[dataframe.SPEC==species_code], date_min, date_max)
    if y == 'DEPTH':
        plot_df.plot(x=x, y=y, kind='scatter', figsize=(30, 12), c='#4C72B0', title=f'{get_species(species_code)}: {y} as a function of {x}').invert_yaxis()
    else:
        plot_df.plot(x=x, y=y, kind='scatter', figsize=(30, 12), c='#4C72B0', title=f'{get_species(species_code)}: {y} as a function of {x}')


############## MAPPING FUNCTIONS ############################


def filter_by_min_species(dataframe, min_species=None):
    """filters dataframe rows where species do not have at least min_species total specimens"""
    if min_species != None:
        spec_counts = pd.DataFrame(dataframe.SPEC.value_counts())
        return dataframe[dataframe['SPEC'].isin(list(spec_counts[spec_counts.SPEC > min_species].index))]
    return dataframe


def _get_species_code_and_name(dataframe, species_code):
    """
    helper function for mapping, 
        converts dataframe and list of species into mappable species codes and names
    species_code='all' returns all species in the dataframe
        the dataframe could be filtered already, eg, filter_by_min_species()
    otherwise, 
        species_code can be an in or a list of ints
        returns name(s) of species and code(s) in list format
    """
    
    if species_code == 'all':
        species_code = list(dataframe.dropna(subset=['SPEC']).SPEC.unique())
        return list(dataframe.dropna(subset=['SPEC']).SPEC.unique()), f'{len(species_code)} Species'
    
    if isinstance(species_code, list):
        species_name = ''
        for i in species_code:
            species_name = species_name + ' / ' + get_species(i)
        species_name = species_name[3:]
    else:
        species_name = get_species(species_code)
        species_code = [species_code]
            
    return species_code, species_name


def filter_by_species(dataframe, species_code):
    """filters dataframe based on inputted species codes ('all', int, or list of ints)"""
    species_code = _get_species_code_and_name(dataframe, species_code)[0]
    return dataframe[dataframe['SPEC'].isin(species_code)]


def _convert_to_geo(dataframe):
    """
    helper function
    filters dataframe by location
    """
    
    # columns to be used in visualisation
    species_mapping_columns = ['DATETIME', 'SPEC', 'COMMON_NAME', 'FLEN', 'FWT', 'MATURITY', 'SEX', 'AGE', 
        'SLAT', 'SLONG', 'ELAT', 'ELONG',  # these are averaged to get LAT, LONG
        'SETNO', 'TOTNO',  # maybe just for debugging
        'STRAT', 'DUR', 'DIST', 'SPEED', 'DEPTH', 'SURF_TEMP', 'BOTT_TEMP', 'BOTT_SAL']  # tow info
        
    # create dataframe with averaged geo, using only the above mapping columns
    map_df = average_geo(dataframe[species_mapping_columns].copy())
    
    # change SPEC to string so that it plots/maps as a discrete categorical variable
    map_df['SPEC'] = map_df['SPEC'].astype(str)

    return map_df


def aggregate_by_geo(dataframe, verbose=False):
    """
    need a geo dataframe with avreaged lat and long
        ie, CALL THE FUNCTION convert_to_geo() first
    min_species
        min_species ignores species with less than min_species total samples
    species_code
        can input species_code='all' for all (filtered by min_species)
        or species_code=int or list of ints for species desired
        can use species_codes_by_percentile()
    """

    aggregation = {
        'DEPTH': ['count', 'max'],  # count is how many fish in that location, max = mean = min (from SETNO not SPEC)
        'FWT': ['sum', 'min', 'max', 'mean'],  # cum is total weight of fish in that location
        'AGE': ['min', 'max', 'mean'], 
        'FLEN': ['min', 'max', 'mean'], 
        'DATETIME': ['mean']
    }

    if verbose:
        aggregation.update({
            'SPEED': ['mean'], 
            'DIST': ['mean'], 
            'DUR': ['mean'], 
            'SURF_TEMP': ['mean'], 
            'BOTT_TEMP': ['mean'], 
            'BOTT_SAL': ['mean']
        })

    gdf = dataframe.drop(['MATURITY', 'SEX', 'STRAT'], axis=1)
    
    return gdf.groupby(['SPEC', 'COMMON_NAME', 'LAT', 'LONG']).agg(aggregation)


def map_species(dataframe, species_code, color='DEPTH', date_min=None, date_max=None, hover_data=None, min_species=None, verbose=False, aggregate_data=True):
    """
    TODO: write a good docstring
    """
        
    # filter by min_species (number in total dataset, filter min number per species first)
    dataframe = filter_by_min_species(dataframe, min_species=min_species)
    
    # filter dataframe by dates
    dataframe = filter_dates(dataframe, date_min=date_min, date_max=date_max)
        
    # filter by species
    dataframe = filter_by_species(dataframe, species_code)
    
    # convert to averaged lat/long and mappable columns
    dataframe = _convert_to_geo(dataframe)
    
    # get the species names(s) for the plot
    species_name = _get_species_code_and_name(dataframe, species_code)[1]
    
    # filter the full dataframe for mapping = dataframe
    if aggregate_data:
        # aggregate by lat/long, species, and filter entries with less than min_species
        dataframe = aggregate_by_geo(dataframe, verbose=verbose)
        dataframe.columns = ['_'.join(col) for col in dataframe.columns.values]  # flatten the dataframe
        dataframe.reset_index(inplace=True)
        dataframe.rename(columns={'DEPTH_count': 'COUNT', 'DEPTH_max': 'DEPTH', 'FWT_sum': 'TOTAL_weight', 'DATETIME_mean': 'DATETIME'}, inplace=True)
    
    # convert DATETIME to year to make it mappable
    dataframe['DATETIME'] = dataframe['DATETIME'].dt.year
        
    # custom hover data if inputted
    if hover_data == None:
        hover_data=dataframe.columns
    else:
        hover_data=hover_data
        
    # make the plot
    fig = px.scatter_geo(dataframe, lat='LAT', lon='LONG', 
        hover_data=hover_data, color=color, 
        color_continuous_scale='Plasma',
        projection='natural earth', scope='north america', 
        title=f'Map of {species_name} Coloured by {color}'
    )
    
    fig.update_geos(resolution=50, projection_scale=8, center=dict(lat=43.5, lon=-63))
    fig.update_layout(
        width=900, height=700, title_x=0.5, 
        margin=dict(l=0, r=0),
        coloraxis=dict(colorbar={'orientation':'h', 'y':0, 'thickness':20})
    )
    
    # display depth below water
    if color == 'DEPTH':
        fig.update_layout(
            coloraxis=dict(
                reversescale = True,
                cauto = False,
                cmin=0, 
                cmax=500
            )
        )

    # show the plot
    fig.show()

