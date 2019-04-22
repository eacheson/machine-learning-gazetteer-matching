'''
@author: eacheson
'''

import pandas as pd
import os

####################
### SWISSNAMES3D ###
####################

class SwissNames3D:
    """
    Represents an instance of the SwissNames3D gazetteer, built from local data.
    """
    def __init__(self, data_dir, verbose=False):
        """
        Constructor, builds the gazetteer from local CSV files.
        :param data_dir: the directory of the SwissNames3D CSV files
        :param verbose: whether to print some output about the building of the instance
        """
        self.data_dir = data_dir
        self.df = self.prepare(verbose)
        self.name_counts = self.df['NAME'].value_counts()
        self.name_groups = self.df.groupby(['NAME'])

    def prepare(self, verbose=False):
        swissnames_csv_filenames = ['swissNAMES3D_' + x + '.csv' for x in ['PKT', 'LIN', 'PLY']]
        # read in all the CSVs
        dfpkt = pd.read_csv(self.data_dir + swissnames_csv_filenames[0], sep=';', encoding='windows-1252', na_values='k_W', low_memory=False)
        dflin = pd.read_csv(self.data_dir + swissnames_csv_filenames[1], sep=';', encoding='windows-1252', na_values='k_W')
        dfply = pd.read_csv(self.data_dir + swissnames_csv_filenames[2], sep=';', encoding='windows-1252', na_values='k_W')
        if verbose:
            print("There are %s point records in the dataset and %s columns" % (dfpkt.shape[0], dfpkt.shape[1]))
            print("There are %s line records in the dataset and %s columns" % (dflin.shape[0], dflin.shape[1]))
            print("There are %s polygon records in the dataset and %s columns" % (dfply.shape[0], dfply.shape[1]))

        # add a column in each dataframe storing the geometry type
        dfpkt['GEOMTYPE'] = 'Point'
        dflin['GEOMTYPE'] = 'Line'
        dfply['GEOMTYPE'] = 'Polygon'

        # concatenate these 3 dataframes together so we can search by name in one go
        df_all = pd.concat([dfpkt, dflin, dfply], join='outer', sort=False)

        # convert the population categories to a sortable int field
        pop_to_int = {"> 100'000": 9, "50'000 bis 100'000": 8, "10'000 bis 49'999": 7,
                      "2'000 bis 9'999": 6, "1'000 bis 1'999": 5, "100 bis 999": 4,
                      "50 bis 99": 3, "20 bis 49": 2, "< 20": 1}

        def calc_pop_to_int(x):
            if x in pop_to_int:
                return pop_to_int[x]
            else:
                return 0

        # population category as int
        df_all['POPCATINT'] = df_all['EINWOHNERKATEGORIE'].apply(calc_pop_to_int)
        if verbose:
            print("The final dataset has %s rows and %s columns" %(df_all.shape[0], df_all.shape[1]))

        # re-order columns in a nicer way
        col_order = ['UUID', 'NAME', 'GEOMTYPE', 'OBJEKTKLASSE_TLM', 'OBJEKTART', 'SPRACHCODE',
                     'E', 'N', 'Z', 'HOEHE', 'EINWOHNERKATEGORIE', 'POPCATINT', 'GEBAEUDENUTZUNG', 'KUNSTBAUTE',
                     'NAMENGRUPPE_UUID', 'NAMEN_TYP', 'NAME_UUID']
        df_all = df_all[col_order]

        return df_all

    def retrieve_by_id(self, uuid, verbose=False):
        """
        Returns the entry corresponding to the uuid passed in.
        :param uuid: the uuid to find the entry for
        :param verbose: set to True to get info printed to screen on method call
        :return: the found entry as a dict, can be empty dict
        """
        try:
            df_match = self.df[self.df['UUID'] == uuid]
            dict_match = df_match.to_dict(orient='records')[0]  # this returns the 'first' match
            if verbose:
                print("Found a match for uuid %s" %uuid)
            return dict_match
        except IndexError:
            if verbose:
                print("Found no match for uuid %s" %uuid)
        return {}

    def exact_match_count(self, name, verbose=False):
        """
        Returns the exact match count for 'name' in the gazetteer.
        :param name: some placename to match exactly
        :param verbose: set to True to get the count printed to screen on method call
        :return: the exact match count for 'name'
        """
        count = 0
        try:
            count = self.name_counts[name]
            if verbose:
                print("Found %s results for %s" % (count, name))
        except KeyError:
            if verbose:
                print("No results found for %s" % name)
        return count

    def string_matches(self, str_token, method='first_word', sort=True, verbose=False):
        """
        Returns matches in the way specified by 'method'.
        :param str_token: word to match in the gazetteer.
        :param method: how to search for string matches; one of 'exact', 'first_word', 'contains_word', 'contains_string';
        defaults to 'first_word'
        :param sort: whether to sort by population category; defaults to True.
        :param verbose: whether to print some output or not
        :return: a pandas dataframe with all the matches, empty dataframe if no matches found.
        """
        df_strm = None
        try:
            if method == 'first_word':
                df_strm = self.first_word_matches(str_token, verbose)
            elif method == 'exact':
                df_strm = self.exact_matches(str_token, verbose)
            elif method == 'contains_word':
                df_strm = self.contains_word_matches(str_token, verbose)
            elif method == 'contains_string':
                df_strm = self.contains_matches(str_token, verbose)
            else:
                print("The matching method specified does not exist.")
                df_strm = pd.DataFrame()
        finally:
            if sort and not df_strm.empty:
                df_sorted = df_strm.sort_values(by='POPCATINT', ascending=False)
                return df_sorted
            return df_strm

    def exact_matches(self, token, verbose=False):
        """
        Returns exact name matches.
        :param token: name to match in the gazetteer.
        :param verbose: whether to print some output or not
        :return: a pandas dataframe with all the exact matches, an empty df if no matches found.
        """
        try:
            df_exact = self.name_groups.get_group(token)
            if verbose:
                print("Found %s exact matches for %s" % (df_exact.shape[0], token))
            return df_exact
        except KeyError:
            if verbose:
                print("Found no exact matches for %s" % token)
        return pd.DataFrame()

    def first_word_matches(self, token, verbose=False):
        """
        Returns matches that start with the full string 'token' as a word, which should not contain spaces.
        :param token: word to match in the gazetteer.
        :param verbose: whether to print some output or not
        :return: a pandas dataframe with all the matches, an empty df if no matches found.
        """
        try:
            df_matches = self.df[self.df['NAME'].str.match('\\b' + token + '\\b', as_indexer=True)]
            if verbose:
                print("Found %s first word matches for %s" % (df_matches.shape[0], token))
            return df_matches
        except KeyError:
            if verbose:
                print("Found no first word matches for %s" % token)
        return pd.DataFrame()

    def contains_word_matches(self, token, verbose=False):
        """
        This methods returns a dataframe with rows that have 'token' appear anywhere in the name string,
        but only as a full word (check for word boundary).
        :param token: string to match in the gazetteer.
        :param verbose: whether to print some output or not
        :return: a pandas dataframe with all the matches, an empty df if no matches found.
        """
        try:
            df_matches = self.df[self.df['NAME'].str.contains('\\b'+token+'\\b')]
            if verbose:
                print("Found %s matches containing word %s" % (df_matches.shape[0], token))
            return df_matches
        except KeyError:
            if verbose:
                print("Found no matches containing word %s" % token)
        return pd.DataFrame()

    def contains_matches(self, token, verbose=False):
        """
        This methods returns a dataframe with rows that have 'token' appear anywhere in the name string.
        Note this is usually far too many matches, as searching for 'Brugg' will return 'Brugg AG', but
        also 'Bruggacker' and 'St. Gallen Bruggen', to name a few.
        :param token: string to match in the gazetteer.
        :param verbose: whether to print some output or not
        :return: a pandas dataframe with all the matches, an empty df if no matches found.
        """
        try:
            df_matches = self.df[self.df['NAME'].str.contains(token)]
            if verbose:
                print("Found %s 'contains' matches for %s" % (df_matches.shape[0], token))
            return df_matches
        except KeyError:
            if verbose:
                print("Found no 'contains' matches for %s" % token)
        return pd.DataFrame()


################
### GEONAMES ###
################

class GeoNamesCH:
    """
    Represents an instance of the GeoNames gazetteer for Switzerland, built from local data.
    """
    def __init__(self, data_dir, verbose=False):
        """
        Constructor, builds the gazetteer from local files.
        :param data_dir: the directory which contains the GeoNames CH.txt file;
        :param verbose: whether to print some output about the building of the instance
        """
        self.data_dir = data_dir
        self.df = self.prepare(verbose)
        self.name_counts = self.df['name'].value_counts() # or asciiname?
        self.name_groups = self.df.groupby(['name'])
        self.type_groups = self.df.groupby(['feature code'])

    def prepare(self, verbose=False):
        colNames = ['geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 'longitude', 'feature class',
                    'feature code', 'country code', 'cc2', 'admin1 code', 'admin2 code', 'admin3 code',
                    'admin4 code', 'population', 'elevation', 'dem', 'timezone', 'modification date']
        df_all = pd.read_csv(os.path.join(self.data_dir, 'CH.txt'), sep='\t', header=None, names=colNames, low_memory=False)
        if verbose:
            print("We have %s records in GeoNames for Switzerland and %s columns." %(df_all.shape[0], df_all.shape[1]))

        return df_all

    def exact_matches(self, token, verbose=False):
        """
        Returns a pandas dataframe.
        :param token: name to match in the gazetteer.
        :param verbose: whether to print some output or not
        :return: a pandas dataframe with all the exact matches, an empty df if no matches found.
        """
        try:
            df_exact = self.name_groups.get_group(token)
            if verbose:
                print("Found %s exact matches for %s" % (df_exact.shape[0], token))
            return df_exact
        except KeyError:
            if verbose:
                print("Found no exact matches for %s" % token)
        return pd.DataFrame()

    def records_of_type(self, type_string, verbose=False):
        """
        Returns a pandas dataframe with all the records matching type_string as the 'feature code'.
        :param type_string: name to match in the gazetteer.
        :param verbose: whether to print some output or not
        :return: a pandas dataframe, an empty df if no matches found.
        """
        try:
            df_exact = self.type_groups.get_group(type_string)
            if verbose:
                print("Found %s records with feature code %s" % (df_exact.shape[0], type_string))
            return df_exact
        except KeyError:
            if verbose:
                print("Found no records with feature code %s" % type_string)
        return pd.DataFrame()

    def retrieve_by_id(self, uid, verbose=False):
        """
        Returns the entry corresponding to the unique id (uid) passed in.
        :param uid: the unique id (uid) to find the entry for
        :param verbose: set to True to get info printed to screen on method call
        :return: the found entry as a dict
        """
        try:
            df_match = self.df[self.df['geonameid'] == uid]
            dict_match = df_match.to_dict(orient='records')[0]
            if verbose:
                print("Found a match for uid %s" %uid)
            return dict_match
        except IndexError:
            if verbose:
                print("Found no match for uid %s" %uid)
        return {}