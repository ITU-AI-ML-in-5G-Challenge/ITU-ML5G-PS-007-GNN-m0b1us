"""
   Copyright 2023 Universitat Politècnica de Catalunya

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

# Only run as main
if __name__ != "__main__":
    raise RuntimeError("This script should not be imported!")

# Parse imports
from typing import Tuple, Generator, Dict, Any, List
import numpy as np
import pandas as pd 
import tensorflow as tf
from itertools import permutations
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import  mutual_info_regression
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from re import sub
import argparse
import sys
import random

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", type=str,  required=True)
parser.add_argument("--shuffle", type=str,  default=True, required=False)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# import datasets's DataNet API
sys.path.insert(0, args.input_dir)
from datanetAPI import DatanetAPI, TimeDist, Sample


def _get_network_decomposition(sample: Sample) -> Tuple[dict, list]:
    """Given a sample from the DataNet API, it returns it as a sample for the model.

    Parameters
    ----------
    sample : DatanetAPI.Sample
        Sample from the DataNet API

    Returns
    -------
    Tuple[dict, list]
        Tuple with the inputs of the model and the target variable to predict

    Raises
    ------
    ValueError
        Raised if one of the links is of an unknown type
    """

    # Read values from the DataNet API
    network_topology = sample.get_physical_topology_object()
    max_link_load = sample.get_max_link_load()
    global_delay = sample.get_global_delay()
    global_losses = sample.get_global_losses()
    traffic_matrix = sample.get_traffic_matrix()
    physical_path_matrix = sample.get_physical_path_matrix()
    performance_matrix = sample.get_performance_matrix()
    packet_info_matrix = sample.get_pkts_info_object()
    # Process sample id (for debugging purposes and for then evaluating the model)
    sample_file_path, sample_file_id = sample.get_sample_id()
    sample_file_name = sample_file_path.split("/")[-1]
    # Obtain links and nodes
    # We discard all links that start from the traffic generator
    links = dict()
    for edge in network_topology.edges:  # src, dst, port
        # We identify all traffic generators as the same port
        edge_id = sub(r"t(\d+)", "tg", network_topology.edges[edge]["port"])
        if edge_id.startswith("r") or edge_id.startswith("s"):
            links[edge_id] = {
                "capacity": float(network_topology.edges[edge].get("bandwidth", 1e9))
                / 1e9,  # original value is in bps, we change it to Gbps
            }
        elif edge_id.startswith("tg"):
            continue
        else:
            raise ValueError(f"Unknown edge type: {edge_id}")

    # In this scenario assume that flows can either follow CBR or MB distributions
    flows = dict()
    used_links = set()  # Used later so we only consider used links
    # Add flows
    for src, dst in filter(
        lambda x: traffic_matrix[x]["AggInfo"]["AvgBw"] != 0
        and traffic_matrix[x]["AggInfo"]["PktsGen"] != 0,
        permutations(range(len(traffic_matrix)), 2),
    ):
        for local_flow_id in range(len(traffic_matrix[src, dst]["Flows"])):
            flow_packet_info = packet_info_matrix[src, dst][0][local_flow_id]
            flow = traffic_matrix[src, dst]["Flows"][local_flow_id]
            # Size distribution is always determinstic
            # Obtain and clean the path followed the flow
            # We must also clean up the name of the traffic generator
            clean_og_path = [
                sub(r"t(\d+)", "tg", link)
                for link in physical_path_matrix[src, dst][2::2]
            ]

            packet_timestamps = np.array([float(x[0]) for x in flow_packet_info])
            ipg = packet_timestamps[1:] - packet_timestamps[-1:]

            flow_id = f"{src}_{dst}_{local_flow_id}"
            flows[flow_id] = {
                "source": src,
                "destination": dst,
                "flow_id": flow_id,
                "length": len(clean_og_path),
                "og_path": clean_og_path,
                "traffic": flow["AvgBw"],  # in bps
                "packets": flow["PktsGen"],
                "flow_variance": flow["VarPktSize"],
                "flow_tos": flow["ToS"],
                "flow_p10PktSize": flow["p10PktSize"],
                "flow_p20PktSize": flow["p20PktSize"],
                "flow_p50PktSize": flow["p50PktSize"],
                "flow_p80PktSize": flow["p80PktSize"],
                "flow_p90PktSize": flow["p90PktSize"],
                "packet_size": flow["SizeDistParams"]["AvgPktSize"],
                "rate": flow["TimeDistParams"]["Rate"] if flow["TimeDist"] == TimeDist.CBR_T else 0,
                "ibg": flow["TimeDistParams"]["IBG"] if flow["TimeDist"] == TimeDist.MULTIBURST_T else 0,
                "flow_bitrate_per_burst": flow["TimeDistParams"]["On_Rate"] if flow["TimeDist"] == TimeDist.MULTIBURST_T else 0,
                "flow_pkts_per_burst": flow["TimeDistParams"]["Pkts_per_burst"] if flow["TimeDist"] == TimeDist.MULTIBURST_T else 0,
                "flow_type": (
                    float(flow["TimeDist"] == TimeDist.CBR_T),
                    float(flow["TimeDist"] == TimeDist.MULTIBURST_T),
                ),
                "delay": performance_matrix[src, dst]["Flows"][local_flow_id]["AvgDelay"] * 1000,  # in ms
                "ipg_mean" : np.mean(ipg) if len(ipg) > 0 else 0,
                "ipg_var": np.var(ipg) if len(ipg) > 0 else 0,  
            }

            # Add edges to the used_links set
            used_links.update(set(clean_og_path))

    # Purge unused links
    links = {kk: vv for kk, vv in links.items() if kk in used_links}

    # Normalize flow naming
    # We give the indices in such a way that flows states are concatanated as [CBR, MB]
    ordered_flows = list()
    flow_mapping = dict()
    for idx, (flow_id, flow_params) in enumerate(flows.items()):
        flow_mapping[flow_id] = idx
        ordered_flows.append(flow_params)
    n_f = len(ordered_flows)

    # Normalize link naming
    ordered_links = list()
    link_mapping = dict()
    for idx, (link_id, link_params) in enumerate(links.items()):
        link_mapping[link_id] = idx
        ordered_links.append(link_params)
    n_l = len(ordered_links)

    # Obtain list of indices representing the topology
    # link_to_path: two dimensional array, first dimension are the paths, second dimension are the link indices
    link_to_path = list()
    # We define link_pos_in_flows that will later help us build path_to_link
    link_pos_in_flows = list()
    for og_path in map(lambda x: x["og_path"], ordered_flows):
        # This list will contain the link indices in the original path,in order
        local_list = list()
        # This dict indicates for each link which are the positions in the original path, if any
        local_dict = dict()
        for link_id in og_path:
            # Transform link_id into a link index
            link_idx = link_mapping[link_id]
            local_dict.setdefault(link_idx, list()).append(len(local_list))
            local_list.append(link_idx)
        link_to_path.append(local_list)
        link_pos_in_flows.append(local_dict)

    # path_to_link: two dimensional array, first dimension are the links, second dimension are tuples.
    # Each tuple contains the path index and the link's position in the path
    # Note that a link can appear in multiple paths and multiple times in the same path
    path_to_link = list()
    for link_idx in range(n_l):
        local_list = list()
        for flow_idx in range(n_f):
            if link_idx in link_pos_in_flows[flow_idx]:
                local_list += [
                    (flow_idx, pos) for pos in link_pos_in_flows[flow_idx][link_idx]
                ]
        path_to_link.append(local_list)

    # Many of the features must have expanded dimensions so they can be concatenated
    sample = (
        {
            # Identifier features

            "sample_file_name": [sample_file_name] * n_f,
            "sample_file_id": [sample_file_id] * n_f,
            "max_link_load": np.expand_dims([max_link_load], axis=1),
            "global_losses": np.expand_dims([global_losses], axis=1),
            "global_delay": np.expand_dims([global_delay], axis=1),
            "flow_id": [flow["flow_id"] for flow in ordered_flows],
            # Flow attributes
            "flow_traffic": np.expand_dims(
                [flow["traffic"] for flow in ordered_flows], axis=1
            ),
            "flow_bitrate_per_burst": np.expand_dims(
                [flow["flow_bitrate_per_burst"] for flow in ordered_flows], axis=1
            ),
            "flow_tos": np.expand_dims(
                [flow["flow_tos"] for flow in ordered_flows], axis=1
            ),
            "flow_p10PktSize": np.expand_dims(
                [flow["flow_p10PktSize"] for flow in ordered_flows], axis=1
            ),
            "flow_p20PktSize": np.expand_dims(
                [flow["flow_p20PktSize"] for flow in ordered_flows], axis=1
            ),
            "flow_p50PktSize": np.expand_dims(
                [flow["flow_p50PktSize"] for flow in ordered_flows], axis=1
            ),       
            "flow_p80PktSize": np.expand_dims(
                [flow["flow_p80PktSize"] for flow in ordered_flows], axis=1
            ),  
            "flow_p90PktSize": np.expand_dims(
                [flow["flow_p90PktSize"] for flow in ordered_flows], axis=1
            ),
            "rate": np.expand_dims(
                [flow["rate"] for flow in ordered_flows], axis=1
            ),
            "ibg": np.expand_dims(
                [flow["ibg"] for flow in ordered_flows], axis=1
            ),
            "flow_variance": np.expand_dims(
                [flow["flow_variance"] for flow in ordered_flows], axis=1
            ),
            "flow_pkts_per_burst": np.expand_dims(
                [flow["flow_pkts_per_burst"] for flow in ordered_flows], axis=1
            ),
            "flow_packets": np.expand_dims(
                [flow["packets"] for flow in ordered_flows], axis=1
            ),
            "flow_packet_size": np.expand_dims(
                [flow["packet_size"] for flow in ordered_flows], axis=1
            ),
            "flow_type": [flow["flow_type"] for flow in ordered_flows],
            "flow_length": [flow["length"] for flow in ordered_flows],
            # Inter-packet gap feature
            "flow_ipg_mean": np.expand_dims(
                [flow["ipg_mean"] for flow in ordered_flows], axis=1
            ),
            "flow_ipg_var": np.expand_dims(
                [flow["ipg_var"] for flow in ordered_flows], axis=1
            ),
            
            # Link attributes
            "link_capacity": np.expand_dims(
                [link["capacity"] for link in ordered_links], axis=1
            ),
            # Topology attributes
            "link_to_path": tf.ragged.constant(link_to_path),
            "path_to_link": tf.ragged.constant(path_to_link, ragged_rank=1),
        },
        [flow["delay"] for flow in ordered_flows],
    )

    return sample


def feature_selection(data_dir: str, shuffle: bool): 

    try:
        data_dir = data_dir.decode("UTF-8")
    except (UnicodeDecodeError, AttributeError):
        pass
    tool = DatanetAPI(data_dir, shuffle=shuffle)
    
    # Subsets of type of features we choose for evaluate
    the_choosen_ones = [ 'ibg', 'flow_packets', 'flow_pkts_per_burst', 'flow_packet_size', 
                        'rate', 'flow_bitrate_per_burst', 'flow_traffic', 'flow_ipg_mean', 
                        'flow_tos', 'flow_length', 'flow_ipg_var',
                        'flow_p10PktSize', 'flow_p20PktSize' , 'flow_p50PktSize' , 
                        'flow_p80PktSize' , 'flow_p90PktSize']

    count = 0
    features_concat = np.array([])
    delay_concat = np.array([])

    for sample in iter(tool):
        flow_features = np.array([])
        ret = _get_network_decomposition(sample)

        # SKIP SAMPLES WITH ZERO OR NEGATIVE VALUES 
        if not all( x > 0 for x in ret[1]):
            continue
        # yield ret
        
        for feature in the_choosen_ones: 
            if feature == 'flow_length':
                ret[0]['flow_length'] = np.expand_dims(ret[0][feature], axis=1)
            if len(flow_features) == 0:
                flow_features = ret[0][feature]
            else:
                flow_features = np.hstack([flow_features, ret[0][feature]])

        ''' Concatenate Features and Labels'''
        if count == 0:
            features_concat = flow_features
            delay_concat = ret[1]
        else:
            features_concat = np.concatenate([features_concat, flow_features])
            delay_concat = np.concatenate([delay_concat, ret[1]])

        count +=1


    print('Perform Standardization across entire dataset')
    mean_label = np.mean(delay_concat)
    mean_feat = np.mean(features_concat)
    
    stand_label = np.std(delay_concat)
    stand_feat = np.std(features_concat)

    norm_label = (delay_concat - mean_label) / stand_label
    norm_feature = (features_concat - mean_feat) / stand_feat

    
    ## Feature Selection
    # ---------------- Wrapper - Exhaustive 
    efs1 = EFS(LinearRegression(), 
           min_features=4,
           max_features=12,
           scoring='neg_mean_squared_error', 
           print_progress=True,
           n_jobs = 9,
           cv = 5)

    efs1 = efs1.fit(norm_feature, norm_label)
    
    best = [efs1.best_idx_]
    type_features_best_mse = []
    print(best)
    for i in best[0] :
        type_features_best_mse.append(the_choosen_ones[i])

    print('Best subset (indices):', efs1.best_idx_ )
    print('Best Score:', efs1.best_score_)
    print('Best subset (corresponding names):', type_features_best_mse)


    # --------------- Filter - Multi Information Gain Method (MIG) 
    mig = mutual_info_regression (norm_feature, norm_label)
    type_features_mig_scores= {}

    for i in range(len(the_choosen_ones)):
        type_features_mig_scores[the_choosen_ones[i]] = mig[i]

    sorted_type_features = sorted(type_features_mig_scores.items(), key=lambda x: x[1], reverse=True)
            
    for type_feature, score in sorted_type_features:
        print("Feature:", type_feature, "-- Score:", score)

# Set seeds for reproducibility
np.random.seed(args.seed)
random.seed(args.seed)
tf.random.set_seed(args.seed)

feature_selection(args.input_dir, args.shuffle)
