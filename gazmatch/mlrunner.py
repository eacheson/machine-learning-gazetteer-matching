'''
@author: eacheson
'''

import pandas as pd
import time
import numpy as np
from leven import levenshtein
import jellyfish
import pyxdameraulevenshtein as pyx
from geopy.distance import vincenty


class FeatureComputer:
    """
    Computes features on our data with the help of external resources. Code is tailored to GeoNames
    as source and SwissNames3D as target, but should be straightforward to adapt (column and index names).
    """
    max_leven_dist = 25

    def __init__(self, matches, gazetteer_source, gazetteer_target, landcover_data=pd.DataFrame(), neigh_obj=None):
        """
        :param matches: dataframe of matched pairs. Computed features get added as columns.
        :param gazetteer_source: gazetteer containing records we want to find matches for
        :param gazetteer_target: gazetteer containing match candidates
        :param neigh_obj: nearest neighbor object for landcover data
        """
        self.df = matches
        self.df_source = gazetteer_source
        self.df_target = gazetteer_target
        self.df_landcover = landcover_data
        self.neigh = neigh_obj

    def calculate_distance(self, row):
        gn_lat = self.df_source.loc[row['geonamesid'], 'latitude']
        gn_lon = self.df_source.loc[row['geonamesid'], 'longitude']
        sn_lat = self.df_target.loc[row['swissnamesid'], 'sn_lat']
        sn_lon = self.df_target.loc[row['swissnamesid'], 'sn_lon']
        # hack for when we have multiple UUID results because of alt name structuring in SwissNames
        if type(sn_lat) == np.ndarray:
            sn_lat = sn_lat[0]
            sn_lon = sn_lon[0]
        v_dist = vincenty([gn_lat, gn_lon], [sn_lat, sn_lon])
        return v_dist.km

    def edit_distance_name(self, row):
        gn_name = self.df_source.loc[row['geonamesid'], 'name']
        sn_name = self.df_target.loc[row['swissnamesid'], 'NAME']
        if type(sn_name) == np.ndarray:
            sn_name = sn_name[0]
        leven_dist = levenshtein(gn_name, sn_name)
        return leven_dist

    def decomma_name(self, name_with_comma):
        parts = name_with_comma.split(",")
        parts_clean = [p.strip() for p in parts]
        if len(parts) == 2:
            return ' '.join(reversed(parts_clean))
        # else
        return name_with_comma

    # for names that are 'flipped' and with a comma, in Geonames (e.g. "Forclaz, Col de la")
    def edit_distance_decomma_name(self, row):
        gn_name = self.decomma_name(self.df_source.loc[row['geonamesid'], 'name'])
        sn_name = self.df_target.loc[row['swissnamesid'], 'NAME']
        if type(sn_name) == np.ndarray:
            sn_name = sn_name[0]
        leven_dist = levenshtein(gn_name, sn_name)
        return leven_dist

    def edit_distance_alt_name(self, row):
        alt_names = str(self.df_source.loc[row['geonamesid'], 'alternatenames'])
        sn_name = self.df_target.loc[row['swissnamesid'], 'NAME']
        if type(sn_name) == np.ndarray:
            sn_name = sn_name[0]
        leven_dist = FeatureComputer.max_leven_dist
        if alt_names != 'nan':
            for altname in alt_names.split(','):
                leven_dist_temp = levenshtein(altname, sn_name)
                if leven_dist_temp < FeatureComputer.max_leven_dist:
                    leven_dist = leven_dist_temp
        return leven_dist

    def jaro_distance(self, row):
        gn_name = self.df_source.loc[row['geonamesid'], 'name']
        sn_name = self.df_target.loc[row['swissnamesid'], 'NAME']
        dist = jellyfish.jaro_distance(gn_name, sn_name)
        return dist

    def jaro_winkler_distance(self, row):
        gn_name = self.df_source.loc[row['geonamesid'], 'name']
        sn_name = self.df_target.loc[row['swissnamesid'], 'NAME']
        dist = jellyfish.jaro_winkler(gn_name, sn_name)
        return dist

    def normalized_leven_dam_distance(self, row):
        gn_name = self.df_source.loc[row['geonamesid'], 'name']
        sn_name = self.df_target.loc[row['swissnamesid'], 'NAME']
        dist = pyx.normalized_damerau_levenshtein_distance(gn_name, sn_name)
        return dist

    def ftype_geonames(self, row):
        source_type = self.df_source.loc[row['geonamesid'], 'feature code']
        return source_type

    def ftype_swissnames(self, row):
        target_type = self.df_target.loc[row['swissnamesid'], 'OBJEKTART']
        if type(target_type) == np.ndarray:
            target_type = target_type[0]
        return target_type

    def elevation_distance(self, row):
        gn_z = self.df_source.loc[row['geonamesid'], 'dem']
        sn_z = self.df_target.loc[row['swissnamesid'], 'Z']
        # take the absolute value
        z_dist = abs(gn_z - sn_z)
        return z_dist

    def elevation_distance_raw(self, row):
        gn_z = self.df_source.loc[row['geonamesid'], 'dem']
        sn_z = self.df_target.loc[row['swissnamesid'], 'Z']
        # just take the difference (raw)
        z_dist = gn_z - sn_z
        return z_dist

    # these landcover functions have to be run with an *augmented* version of df_source that has swiss coordinates
    def landcover_geonames(self, row):
        E = self.df_source.loc[row['geonamesid'], 'gn_E']
        N = self.df_source.loc[row['geonamesid'], 'gn_N']
        cell = self.neigh.kneighbors(X=[[E, N]], n_neighbors=1, return_distance=False)
        index = cell[0][0]
        neighbour = self.df_landcover.iloc[index, :]
        source_landcover = neighbour['LC09_6']
        return source_landcover

    def landcover_swissnames(self, row):
        E = self.df_target.loc[row['swissnamesid'], 'E']
        N = self.df_target.loc[row['swissnamesid'], 'N']
        cell = self.neigh.kneighbors(X=[[E, N]], n_neighbors=1, return_distance=False)
        index = cell[0][0]
        neighbour = self.df_landcover.iloc[index, :]
        target_landcover = neighbour['LC09_6']
        return target_landcover

    def landcover_geonames_mode(self, row):
        E = self.df_source.loc[row['geonamesid'], 'gn_E']
        N = self.df_source.loc[row['geonamesid'], 'gn_N']
        # get 9 NN because that's 2 grid layers
        cell = self.neigh.kneighbors(X=[[E, N]], n_neighbors=9, return_distance=False)
        indices = cell[0]
        neighbour_subset = self.df_landcover.iloc[indices, :]
        # get the 'mode' landcover class
        source_landcover = neighbour_subset['LC09_6'].value_counts().idxmax()
        return source_landcover

    def landcover_swissnames_mode(self, row):
        E = self.df_target.loc[row['swissnamesid'], 'E']
        N = self.df_target.loc[row['swissnamesid'], 'N']
        # get 9 NN because that's 2 grid layers
        cell = self.neigh.kneighbors(X=[[E, N]], n_neighbors=9, return_distance=False)
        indices = cell[0]
        neighbour_subset = self.df_landcover.iloc[indices, :]
        # get the 'mode' landcover class
        target_landcover = neighbour_subset['LC09_6'].value_counts().idxmax()
        return target_landcover

    def landcover_distance(self, row):
        gn_E = self.df_source.loc[row['geonamesid'], 'gn_E']
        gn_N = self.df_source.loc[row['geonamesid'], 'gn_N']
        E = self.df_target.loc[row['swissnamesid'], 'E']
        N = self.df_target.loc[row['swissnamesid'], 'N']
        # geonames vector
        gn_cell = self.neigh.kneighbors(X=[[gn_E, gn_N]], n_neighbors=9, return_distance=False)
        gn_indices = gn_cell[0]
        gn_neighbour_subset = self.df_landcover.iloc[gn_indices, :]
        gn_vc_dict = gn_neighbour_subset['LC09_6'].value_counts().to_dict()
        gn_l = [gn_vc_dict[x] if x in gn_vc_dict else 0 for x in range(10, 61, 10)]
        # swissnames vector
        cell = self.neigh.kneighbors(X=[[E, N]], n_neighbors=9, return_distance=False)
        indices = cell[0]
        neighbour_subset = self.df_landcover.iloc[indices, :]
        vc_dict = neighbour_subset['LC09_6'].value_counts().to_dict()
        l = [vc_dict[x] if x in vc_dict else 0 for x in range(10, 61, 10)]
        vector_diff = np.array(gn_l) - np.array(l)
        lc_distance = np.absolute(vector_diff).sum()
        return lc_distance

    def compute_features_all(self, verbose=False):
        """
        Compute all the features. Later we can choose which ones to actually
        use to train a model in order to test different feature combinations.
        """
        if verbose:
            print("Calculating point-to-point distance between matches...")
        time1 = time.time()
        self.df['distance'] = self.df.apply(self.calculate_distance, axis=1)
        if verbose:
            print("Calculating Levenshtein distance between names...")
        self.df['leven'] = self.df.apply(self.edit_distance_name, axis=1)
        if verbose:
            print("Calculating Levenshtein distance between de-commaed names...")
        self.df['leven_comma'] = self.df.apply(self.edit_distance_decomma_name, axis=1)
        if verbose:
            print("Calculating min Levenshtein distance on alternate names...")
        self.df['leven_alt'] = self.df.apply(self.edit_distance_alt_name, axis=1)
        if verbose:
            print("Getting the min Levenshtein distance overall...")
        self.df['leven_min'] = self.df.apply(lambda r: min(r['leven'], r['leven_comma'], r['leven_alt']), axis=1)
        if verbose:
            print("Calculating the normalized Levenshtein-Damerau distance...")
        self.df['leven_dam_norm'] = self.df.apply(self.normalized_leven_dam_distance, axis=1)
        if verbose:
            print("Calculating Jaro similarity...")
        self.df['jaro'] = self.df.apply(self.jaro_distance, axis=1)
        if verbose:
            print("Calculating Jaro-Winkler similarity...")
        self.df['jaro_winkler'] = self.df.apply(self.jaro_winkler_distance, axis=1)
        if verbose:
            print("Getting the absolute elevation distance...")
        self.df['elev_dist'] = self.df.apply(self.elevation_distance, axis=1)
        if verbose:
            print("Getting the feature types from both gazetteers...")
        self.df['type_geonames'] = self.df.apply(self.ftype_geonames, axis=1)
        self.df['type_swissnames'] = self.df.apply(self.ftype_swissnames, axis=1)
        if verbose:
            print("Getting the dummy variables for feature types...")
        self.df = pd.get_dummies(data=self.df, columns=['type_geonames', 'type_swissnames'], prefix=['gn', 'sn'])
        if verbose:
            print("Getting the landcover classes for source and target...")
        self.df['gn_landcover'] = self.df.apply(self.landcover_geonames, axis=1)
        self.df['sn_landcover'] = self.df.apply(self.landcover_swissnames, axis=1)
        if verbose:
            print("Getting the dummy variables for landcover classes...")
        self.df = pd.get_dummies(data=self.df, columns=['gn_landcover', 'sn_landcover'], prefix=['gn', 'sn'])
        if verbose:
            print("Getting the 'mode' landcover classes for source and target...")
        self.df['gn_landcover_mode'] = self.df.apply(self.landcover_geonames_mode, axis=1)
        self.df['sn_landcover_mode'] = self.df.apply(self.landcover_swissnames_mode, axis=1)
        if verbose:
            print("Getting the dummy variables for 'mode' landcover classes...")
        self.df = pd.get_dummies(data=self.df, columns=['gn_landcover_mode', 'sn_landcover_mode'],
                                 prefix=['gn_mode', 'sn_mode'])
        if verbose:
            print("Getting the landcover 'distance'...")
        self.df['lc_distance'] = self.df.apply(self.landcover_distance, axis=1)
        time2 = time.time()
        if verbose:
            print('Feature computation took %0.3fs' % (time2 - time1))


class CandidateSelecter:
    """
    Class to look for candidate record pairs using various strategies. The candidate pairs should
    ideally include the true matches, since in a realistic scenario, this is how one would go about
    selecting a subset of the very large number of possible pairs for classification as 'match' or 'no-match'.
    """
    max_leven_dist = 25

    def __init__(self, gazetteer_source, gazetteer_target, neigh_obj=None):
        """
        :param gazetteer_source: gazetteer containing records we want to find matches for
        :param gazetteer_target: gazetteer containing match candidates
        :param neigh_obj: nearest neighbors object for target gazetteer
        """
        self.df_source = gazetteer_source
        self.df_target = gazetteer_target
        self.target_name_groups = self.df_target.groupby(['NAME'])
        self.soft_type_align = self.make_soft_align()
        self.neigh = neigh_obj

    def make_soft_align(self):
        soft_align = {}
        soft_align['LK'] = ['Seeteil', 'See']
        soft_align['GLCR'] = ['Gletscher', 'Alpiner Gipfel']
        soft_align['STM'] = ['Fliessgewaesser']
        soft_align['PK'] = ['Grat', 'Huegelzug', 'Hauptgipfel', 'Gipfel',
                            'Huegel', 'Haupthuegel', 'Felskopf', 'Alpiner Gipfel']
        soft_align['PASS'] = ['Pass', 'Graben']
        soft_align['HLL'] = ['Grat', 'Huegelzug', 'Hauptgipfel', 'Gipfel', 'Huegel',
                             'Haupthuegel', 'Felskopf', 'Alpiner Gipfel']
        soft_align['MT'] = ['Grat', 'Huegelzug', 'Hauptgipfel', 'Gipfel', 'Huegel',
                            'Haupthuegel', 'Felskopf', 'Alpiner Gipfel']
        soft_align['VAL'] = ['Tal', 'Haupttal', 'Graben']
        return soft_align

    def get_random_candidates(self, id_list, matches_per_record):
        """
        Get a specified number of random matches for each source id.
        :param id_list: list of source ids
        :param matches_per_record: how many potential matches to keep per source record
        :return: potential matches dict (mix of positives and negatives)
        """
        potential_matches = {}
        for source_id in id_list:
            # random entry in our target gazetteer for each match
            df_candidates = self.df_target.sample(n=matches_per_record)
            # probably all of these will be true negatives, but by chance we could get some positives
            potential_matches[source_id] = df_candidates['UUID'].tolist()
        return potential_matches

    # this relies on an *augmented* version of df_source that has swiss coordinates
    def get_candidates_geodistance(self, id_list, matches_per_record, verbose=False):
        """
        Keep pairs that score highly on point-to-point distance (i.e. low distances). Here we use
        the nearest neighbors object with euclidean distance to choose the k target records closest
        to our source record, for each record.
        :param id_list: list of source ids
        :param matches_per_record: how many potential matches to keep per record
        :return: potential matches dict (mix of positives and negatives)
        """
        counter = 0
        potential_matches = {}
        for source_id in id_list:
            if verbose:
                if (counter % 50) == 0:
                    print("Processing record %s..." % counter)
            E = self.df_source.loc[source_id, 'gn_E']
            N = self.df_source.loc[source_id, 'gn_N']
            # we could keep the distance here... but we also might need to filter this for positives
            cell = self.neigh.kneighbors(X=[[E, N]], n_neighbors=matches_per_record, return_distance=False)
            indices = cell[0]
            neighbour_subset = self.df_target.iloc[indices, :]
            potential_matches[source_id] = neighbour_subset['UUID'].tolist()
            counter += 1
        return potential_matches

    def get_candidates_levenshtein(self, id_list, matches_per_record, verbose=False):
        """
        Keep pairs that score highly on levenshtein distance (i.e. low distances).
        :param id_list: list of source ids
        :param matches_per_record: how many potential matches to keep per record
        :return: potential matches dict (mix of positives and negatives)
        """
        counter = 0
        potential_matches = {}
        for source_id in id_list:
            if verbose:
                if (counter % 50) == 0:
                    print("Processing record %s..." % counter)
            s_name = self.df_source.loc[source_id, 'name']
            s_altnames = str(self.df_source.loc[source_id, 'alternatenames'])
            s_type = self.df_source.loc[source_id, 'feature code']

            # first get any exact name matches as candidates (of any feature type)
            df_exact = self.find_all_exact_name_matches(s_name, s_altnames, self.target_name_groups)
            df_exact['leven_dist'] = 0
            # consider only records of compatible types for the rest
            df_candidates = self.df_target[self.df_target['OBJEKTART'].isin(self.soft_type_align[s_type])].copy()
            # calculate levenshtein distance for just feature-type candidates
            df_candidates['leven_dist'] = df_candidates.apply(lambda r: self.get_edit_distance(r, s_name), axis=1)
            df_candidates = df_candidates.append(df_exact)
            df_candidates = df_candidates.drop_duplicates()
            df_candidates_sorted = df_candidates.sort_values(by='leven_dist', ascending=True)
            # here we keep the top k matches, regardless of whether positive or negative
            potential_matches[source_id] = df_candidates_sorted.iloc[:matches_per_record]['UUID'].tolist()
            counter += 1
        return potential_matches

    def get_candidates_geoleven(self, id_list, matches_per_record):
        """
        Keep the top k matches sorted according to a weighted sum of
        levenshtein distance (on main or alt name) and point-to-point distance.
        :param id_list: list of source ids
        :param matches_per_record: how many potential matches to keep per record
        :return: potential matches dict (mix of positives and negatives)
        """
        counter = 0
        potential_matches = {}
        for source_id in id_list:
            if (counter % 50) == 0:
                print("Processing record %s..." % counter)
            s_name = self.df_source.loc[source_id, 'name']
            s_altnames = str(self.df_source.loc[source_id, 'alternatenames'])
            s_lat = self.df_source.loc[source_id, 'latitude']
            s_lon = self.df_source.loc[source_id, 'longitude']
            s_type = self.df_source.loc[source_id, 'feature code']
            s_point = [s_lat, s_lon]

            # first get any exact name matches as candidates (of any feature type)
            df_exact = self.find_all_exact_name_matches(s_name, s_altnames, self.target_name_groups)
            df_exact['leven_dist'] = 0
            # consider only records of compatible types for the rest
            df_candidates = self.df_target[self.df_target['OBJEKTART'].isin(self.soft_type_align[s_type])].copy()
            # calculate levenshtein distance for just feature-type candidates
            df_candidates['leven_dist'] = df_candidates.apply(lambda r: self.get_edit_distance(r, s_name), axis=1)
            # combine then calculate geo distance for all candidates so far
            df_candidates = df_candidates.append(df_exact)
            df_candidates['distances'] = df_candidates.apply(lambda r: self.calculate_distance_local(r, s_point), axis=1)
            # combine text distance with geodistance for a total score
            df_candidates['similarity'] = df_candidates.apply(self.similarity_points, axis=1)
            df_candidates = df_candidates.drop_duplicates()
            df_candidates_sorted = df_candidates.sort_values(by='similarity', ascending=True)
            # here we keep the top k matches, regardless of whether positive or negative
            potential_matches[source_id] = df_candidates_sorted.iloc[:matches_per_record]['UUID'].tolist()

            # OPTIONAL: also take M random records from SwissNames3D per geonames record
            # df_neg_candidates = swissnames.sample(n=neg_per_pos)
            # candidate_list = df_neg_candidates['UUID'].tolist()
            counter += 1
        return potential_matches

    def get_candidates_geoleven_prefilter(self, id_list, matches_per_record, max_distance=50000, filter_by_type=False):
        """
        Keep the top k matches sorted according to a weighted sum of levenshtein distance (on
        main or alt name) and point-to-point distance. A filter on non-exact name matches is done
        spatially, discarding any feature further away than the distance specified by max_distance
        (in metres), and optionally a type filter is also applied after the spatial filter.
        and
        :param id_list: list of source ids
        :param matches_per_record: how many potential matches to keep per record
        :param max_distance: discard records further away than this distance (in metres)
        :return: potential matches dict (mix of positives and negatives)
        """
        counter = 0
        potential_matches = {}
        for source_id in id_list:
            if (counter % 50) == 0:
                print("Processing record %s..." % counter)
            s_name = self.df_source.loc[source_id, 'name']
            s_altnames = str(self.df_source.loc[source_id, 'alternatenames'])
            s_lat = self.df_source.loc[source_id, 'latitude']
            s_lon = self.df_source.loc[source_id, 'longitude']
            s_type = self.df_source.loc[source_id, 'feature code']
            s_E = self.df_source.loc[source_id, 'gn_E']
            s_N = self.df_source.loc[source_id, 'gn_N']
            s_point = [s_lat, s_lon]

            # spatial filter first
            cell = self.neigh.radius_neighbors(X=[[s_E, s_N]], radius=max_distance)
            distances = cell[0][0]
            indices = cell[1][0]
            df_candidates = self.df_target.iloc[indices, :].copy()
            df_candidates['distances'] = distances
            df_candidates['leven_dist'] = df_candidates.apply(lambda r: self.get_edit_distance(r, s_name), axis=1)

            # optional type filter
            if filter_by_type:
                df_candidates = df_candidates[df_candidates['OBJEKTART'].isin(self.soft_type_align[s_type])].copy()

            # include exact name matches no matter what
            df_exact = self.find_all_exact_name_matches(s_name, s_altnames, self.target_name_groups)
            if not df_exact.empty:
                df_exact['distances'] = df_exact.apply(lambda r: self.calculate_distance_metres(r, s_point), axis=1)
                df_exact['leven_dist'] = 0
                df_candidates = df_candidates.append(df_exact)

            if not df_candidates.empty:
                df_candidates['similarity'] = df_candidates.apply(self.similarity_points, axis=1)
                df_candidates = df_candidates.drop_duplicates()
                df_candidates_sorted = df_candidates.sort_values(by='similarity', ascending=True)
                # here we keep the top k matches, regardless of whether positive or negative
                potential_matches[source_id] = df_candidates_sorted.iloc[:matches_per_record]['UUID'].tolist()
            else:
                potential_matches[source_id] = []

            counter += 1
        return potential_matches

    def decomma_name(self, name_with_comma):
        parts = name_with_comma.split(",")
        parts_clean = [p.strip() for p in parts]
        if len(parts) == 2:
            return ' '.join(reversed(parts_clean))
        # else
        return name_with_comma

    def get_exact_matches(self, token, grouped_df, df_cols):
        try:
            df_ret = grouped_df.get_group(token)
            return df_ret
        except KeyError:
            return pd.DataFrame(columns=df_cols)

    def find_all_exact_name_matches(self, name, altnames, name_groups):
        # simple exact matches
        df_ret = self.get_exact_matches(name, name_groups, self.df_target.columns)
        # exact matches on flipped comma name
        df_ret = df_ret.append(self.get_exact_matches(self.decomma_name(name), name_groups, self.df_target.columns))
        # alternate name exact matches
        if altnames != 'nan':
            for altname in altnames.split(','):
                df_ret = df_ret.append(self.get_exact_matches(altname, self.target_name_groups, self.df_target.columns))
        return df_ret

    # for use with the augmented swissnames file (pre-projected to lat lon)
    def calculate_distance_local(self, row, point):
        point_proj = [row['sn_lat'], row['sn_lon']]
        v_dist = vincenty(point, point_proj)
        return v_dist.km

    # for our spatial filtering candidate selection
    def calculate_distance_metres(self, row, point):
        point_proj = [row['sn_lat'], row['sn_lon']]
        v_dist = vincenty(point, point_proj)
        return v_dist.m

    # for use on a row from swissnames, and a geonames name ('geoname')
    def get_edit_distance(self, row, geoname):
        sn_name = row['NAME']
        leven_dist = levenshtein(geoname, sn_name)
        return leven_dist

    def similarity_points(self, row):
        weight1 = 0.5
        weight2 = 0.5
        sim_score = weight1 * (row['leven_dist'] ** 2) + weight2 * (row['distances'] ** 2)
        return sim_score
