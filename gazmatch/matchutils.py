'''
@author: eacheson
'''

import pickle
import pandas as pd
from geopy.distance import vincenty
from sklearn.neighbors import NearestNeighbors

# GLOBAL strings
NO_MATCH_STRING = "no-match"
MATCH_STRING = "match"

### Rule-based matching functions ###

# for use with the augmented swissnames file (pre-projected to lat lon)
def calculate_distance_local(row, point):
    point_proj = [row['sn_lat'], row['sn_lon']]
    v_dist = vincenty(point, point_proj)
    return v_dist.km

def get_exact_matches(token, grouped_df, df_cols):
    try:
        df_ret = grouped_df.get_group(token)
        return df_ret
    except KeyError:
        return pd.DataFrame(columns=df_cols)

def decomma_name(name_with_comma):
    """
    Process names that are 'flipped' and with a comma, in Geonames (e.g. "Forclaz, Col de la")
    :param name_with_comma: original name
    :return: flipped name
    """
    parts = name_with_comma.split(",")
    parts_clean = [p.strip() for p in parts]
    if len(parts) == 2:
        return ' '.join(reversed(parts_clean))
    # else
    return name_with_comma

def find_all_exact_name_matches(name, altnames, name_groups, df_target):
    # simple exact matches
    df_ret = get_exact_matches(name, name_groups, df_target.columns)
    # exact matches on flipped comma name
    df_ret = df_ret.append(get_exact_matches(decomma_name(name), name_groups, df_target.columns))
    # alternate name exact matches
    if altnames != 'nan':
        for altname in altnames.split(','):
            df_ret = df_ret.append(get_exact_matches(altname, name_groups, df_target.columns))
    df_ret = df_ret.drop_duplicates()
    return df_ret


### Data preparation and handling ###

def load_data(pathToPickleFile):
    with open(pathToPickleFile, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    return data

def pickle_data(data, pathToPickleFile):
    with open(pathToPickleFile, 'wb') as pickle_file:
        pickle.dump(data, pickle_file)
    print("pickled data at " + pathToPickleFile)
    return True

def new_uuid_to_old(new_uuid):
    """
    Recover SwissNames3D geographic feature UUID ('old UUID') from custom record-unique IDs ('new UUID').
    :param new_uuid: the new UUID
    :return: the shorter swissnames UUID
    """
    sep = "}"
    old_uuid = new_uuid.split(sep)[0] + sep
    return old_uuid

def old_uuid_to_new(name, uuid, lang_id):
    """
    Makes a custom record-unique ID ('new UUID') using a SwissNames3D UUID ('old UUID'), by concatenating this UUID
    with the name (no space) and a 2-letter language code.
    :param name: name of the feature
    :param uuid: swissnames UUID of the feature
    :param lang_id: 2-letter language code ('de', 'fr', 'it', 'rm') or 'me' for multi-lingual and 'na' for NA/other
    :return: a new record-unique ID made from combining the parameters
    """
    name_no_spaces = name.replace(' ', '')
    new_uuid = uuid + '-' + name_no_spaces + '-' + lang_id
    return new_uuid

def prepare_neighbours(landcover_data):
    # prepare data as x, y coordinates
    columns = ['X', 'Y']
    LC = landcover_data[columns]
    # make a nearest neighbors object:
    # n_neighbors=9 as default, bc if we think of a grid it would be 8+1 for cell+surrounding
    nn_obj = NearestNeighbors(n_neighbors=9, algorithm='kd_tree', metric='euclidean')
    # fit to our sample data (this takes a while!)
    nn_obj.fit(LC)
    return nn_obj

def prepare_neighbours_swissnames(swissnames_data):
    # use the East and North columns, which are in swiss coord and meter units
    columns = ['E', 'N']
    SN = swissnames_data[columns]
    # make a nearest neighbors object, lots of potential parameters here
    nn_obj = NearestNeighbors(n_neighbors=10, algorithm='kd_tree', metric='euclidean')
    # fit to our sample data (this takes a while!)
    nn_obj.fit(SN)
    return nn_obj

def prepare_positives_from_ids(source_ids, gt_dict, verbose=False):
    """
    Prepare positive matches: starts from a list of source ids and grabs any positive
    matches from the ground truth dictionary, returning a flattened data structure
    (dataframe) of positive matches.
    :param source_ids: list of source ids
    :param gt_dict: ground truth dictionary
    :param verbose: print some output during method call; defaults to False
    :return: a dataframe of positive match pairs
    """
    positive_matches = []
    no_match = 0
    for gn_id in source_ids:
        sn_ids = gt_dict[gn_id]
        if not sn_ids:
            # list is empty
            no_match += 1
        for sn_id in sn_ids:
            positive_matches.append((gn_id, sn_id, MATCH_STRING))

    if verbose:
        print("We have %s positive matches and %s records with no match" % (
        len(positive_matches), no_match))

    # turn positive matches into a dataframe
    df_positives = pd.DataFrame.from_records(positive_matches, columns=['geonamesid', 'swissnamesid', 'match'])
    return df_positives

def prepare_positives(train_ids, test_ids, gt_dict, verbose=False):
    """
    Prepare positive matches: starts from a list of source training and testing ids,
    and grabs any positive matches from the ground truth dictionary, returning a flattened data
    structure of positive matches, one dataframe for training and one for testing.
    :param train_ids: list of source ids in training set
    :param test_ids: list of source ids in test set
    :param gt_dict: ground truth dictionary
    :param verbose: print some output during method call; defaults to False
    :return: two dataframes, first the training positives, then the testing positives
    """
    positives_train = []
    no_match_train = 0
    for gn_id in train_ids:
        sn_ids = gt_dict[gn_id]
        if not sn_ids:
            # list is empty
            no_match_train += 1
        for sn_id in sn_ids:
            positives_train.append((gn_id, sn_id, MATCH_STRING))

    positives_test = []
    no_match_test = 0
    for gn_id in test_ids:
        sn_ids = gt_dict[gn_id]
        if not sn_ids:
            # list is empty
            no_match_test += 1
        for sn_id in sn_ids:
            positives_test.append((gn_id, sn_id, MATCH_STRING))

    if verbose:
        print("Training: We have %s positive matches and %s records with no match" % (len(positives_train), no_match_train))
        print("Testing: We have %s positive matches and %s records with no match" % (len(positives_test), no_match_test))

    # turn positive matches into a dataframe, for train and test
    df_positives_train = pd.DataFrame.from_records(positives_train, columns=['geonamesid', 'swissnamesid', 'match'])
    df_positives_test = pd.DataFrame.from_records(positives_test, columns=['geonamesid', 'swissnamesid', 'match'])
    return df_positives_train, df_positives_test

def split_pos_neg(matches_dict, gold_dict, verbose=False):
    """
    Split match candidates (dict) into positive and negative records (flat), based on
    dict of ground truth.
    :param matches_dict: potential matches found by algorithm {gn_id:[list_of_sn_ids]}
    :param gold_dict: ground truth matches (one to 0, one to 1, or one to many)
    :return: a list of positive matches, and a list of negative matches
    """
    pos_records = []
    neg_records = []
    pos = 0
    neg = 0
    for gn_id, sn_ids in matches_dict.items():
        gold_matches = gold_dict[gn_id]
        if not gold_matches:
            # 'no match' is ground truth, therefore all ids found are negative matches
            neg += len(sn_ids)
            for sn_id in sn_ids:
                neg_records.append((gn_id, sn_id, NO_MATCH_STRING))
        else:
            # go through ids found
            for sn_id in sn_ids:
                if sn_id in gold_matches:
                    pos += 1
                    pos_records.append((gn_id, sn_id, MATCH_STRING))
                else:
                    neg += 1
                    neg_records.append((gn_id, sn_id, NO_MATCH_STRING))
    if verbose:
        if pos > 0:
            print("Summary:\n  %s potential matches\n  %s positive matches\n  %s negative matches\n  ratio of %.2f" %(pos+neg, pos, neg, neg/pos))
        else:
            print("Summary:\n  %s potential matches\n  %s positive matches\n  %s negative matches" %(pos+neg, pos, neg))
    return pos_records, neg_records


### Evaluation functions ###

def calculate_recall_upper_bound(test_unique_positives, df_test_all_positives):
    pos_pairs_old_uuid = [(item[0], new_uuid_to_old(item[1])) for item in test_unique_positives]
    pos_pairs_gt = df_test_all_positives[['geonamesid', 'swissnamesid']].copy()
    pos_pairs_gt_list = [tuple(x) for x in pos_pairs_gt.values]
    pos_pairs_gt_list_old_uuid = [(item[0], new_uuid_to_old(item[1])) for item in pos_pairs_gt_list]
    # positive pairs in retained candidates
    matches_set = set(pos_pairs_old_uuid)
    # positive pairs in ground truth
    gt_set = set(pos_pairs_gt_list_old_uuid)
    # set logic
    tp_set = matches_set.intersection(gt_set)
    recall_upper_bound = len(tp_set) / len(gt_set)
    return recall_upper_bound

def get_positives_as_pairs(test_df, preds_col):
    df_results = test_df.copy()
    df_results['predicted'] = preds_col
    positives = df_results[df_results['predicted'] == 0]
    #print(positives.shape)
    pos_pairs = positives[['geonamesid', 'swissnamesid']].copy()
    #print(pos_pairs_b.shape)
    pos_pair_list = [tuple(x) for x in pos_pairs.values]
    return pos_pair_list

def filter_ground_truth_for_test(gt_list_full, source_records_test, verbose=False):
    gt_list_filtered = [item for item in gt_list_full if item[0] in source_records_test]
    if verbose:
        print("full: %s pairs" %len(gt_list_full))
        print("filtered: %s pairs" %len(gt_list_filtered))
    return gt_list_filtered

def evaluate_results_deep(matched_pairs, gt_pairs, verbose=False):
    # positive pairs predicted
    matches_set = set(matched_pairs)
    # positive pairs in ground truth
    gt_set = set(gt_pairs)
    # use set logic
    tp_set = matches_set.intersection(gt_set)
    precision = len(tp_set) / len(matches_set)
    recall = len(tp_set) / len(gt_set)
    f1 = (2.0 * precision * recall) / (precision + recall)
    if verbose:
        print("  %.3f precision\n  %.3f recall\n  %.3f f1" %(precision, recall, f1))
    return precision, recall, f1

def evaluate_results_list(matches_list, negative_list, ground_truth_list, verbose=True):
    """
    Function to evaluate results for rule-based matching, using a list of ground truth pairs
    and some set logic for precision, recall.
    :param matches_list: list of tuples of matching IDs [(gn_id, sn_id)]
    :param negative_list: list of tuples predicted as 'no-match' [(gn_id, 'no-match')]
    :param ground_truth_list: list of tuples of ground truth matching IDs [(gn_id, sn_id)]
    :param verbose: whether to print output or not
    :return: precision, recall, deep_recall, f1, deep_f1, accuracy
    """
    # initial check: all lists should actually be sets
    matches_set = set(matches_list)
    negative_set = set(negative_list)
    ground_truth_set = set(ground_truth_list)
    # for debugging, make this True
    if False:
        if len(matches_set) != len(matches_list):
            print("matches list had duplicates")
        if len(negative_set) != len(negative_list):
            print("negatives list had duplicates")
        if len(ground_truth_set) != len(ground_truth_list):
            print("ground truth list had duplicates")

    # precision: correctly_guessed_matches/total_guessed_matches
    tp_set = matches_set.intersection(ground_truth_set)
    precision = len(tp_set) / len(matches_set)

    # recall: correctly_guessed_matches/total_matches
    recall = len(tp_set) / len(ground_truth_set)

    # F1: based only on precision and recall
    f1 = (2.0 * precision * recall) / (precision + recall)

    if verbose:
        print("%.3f\t%.3f\t%.3f" % (precision, recall, f1))

    return precision, recall, f1
