import pygraphviz as pgv
import itertools
from collections import defaultdict
from typing import Dict, Set
from PIL import Image, ImageOps
from collections import Counter
from ipywidgets import interact
import ipywidgets as widgets
import csv
import pandas as pd
import pm4py
from collections import defaultdict
import numpy as np
from collections import Counter
from io import StringIO
import streamlit as st
from tempfile import *
import os

def get_causality(direct_succession) -> Dict[str, Set[str]]:
    causality = defaultdict(set)
    for ev_cause, events in direct_succession.items():
        for event in events:
            if ev_cause not in direct_succession.get(event, set()):
                causality[ev_cause].add(event)
    return dict(causality)
def get_inv_causality(causality) -> Dict[str, Set[str]]:
    inv_causality = defaultdict(set)
    for key, values in causality.items():
        for value in values:
            inv_causality[value].add(key)
    return {k: v for k, v in inv_causality.items() if len(v) > 1}



class MyGraph(pgv.AGraph):
    def __init__(self, *args):
        super(MyGraph, self).__init__(strict=False, directed=True, *args)
        self.graph_attr['rankdir'] = 'LR'
        self.node_attr['shape'] = 'Mrecord'
        self.graph_attr['splines'] = 'ortho'
        self.graph_attr['nodesep'] = '0.8'
        self.edge_attr.update(penwidth='2')
        self.multigateways = set()

    def add_event(self, name):
        super(MyGraph, self).add_node(name, shape="circle", label="")

    def add_end_event(self, name):
        super(MyGraph, self).add_node(name, shape="circle", label="",penwidth='3')

    def add_and_gateway(self, *args):
        super(MyGraph, self).add_node(*args, shape="diamond",
                                  width=".7",height=".7",
                                  fixedsize="true",
                                  fontsize="40",label="+")

    def add_xor_gateway(self, *args, **kwargs):
        super(MyGraph, self).add_node(*args, shape="diamond",
                                  width=".7",height=".7",
                                  fixedsize="true",
                                  fontsize="40",label="×")

    def add_and_split_gateway(self, source, targets, *args):
        gateway = 'ANDs '+str(source)+'->'+str(targets)
        self.add_and_gateway(gateway,*args)
        super(MyGraph, self).add_edge(source, gateway)
        for target in targets:
            super(MyGraph, self).add_edge(gateway, target)
        return gateway

    def add_xor_split_gateway(self, source, targets, *args):
        gateway = 'XORs '+str(source)+'->'+str(targets)
        self.add_xor_gateway(gateway, *args)
        super(MyGraph, self).add_edge(source, gateway)
        for target in targets:
            super(MyGraph, self).add_edge(gateway, target)
        return gateway

    def add_and_merge_gateway(self, sources, target, *args):
        gateway = 'ANDm '+str(sources)+'->'+str(target)
        self.add_and_gateway(gateway,*args)
        super(MyGraph, self).add_edge(gateway,target)
        for source in sources:
            super(MyGraph, self).add_edge(source, gateway)
        return gateway

    def add_xor_merge_gateway(self, sources, target, *args):
        gateway = 'XORm '+str(sources)+'->'+str(target)
        self.add_xor_gateway(gateway, *args)
        super(MyGraph, self).add_edge(gateway,target)
        for source in sources:
            super(MyGraph, self).add_edge(source, gateway)
        return gateway

    def connect_xor_to_and(self, sources, targets, *args):
        xor_merge_gateway = 'XORm '+str(sources)+'->'+str(targets)
        and_split_gateway = 'ANDs '+str(sources)+'->'+str(targets)
        multigateway = and_split_gateway + xor_merge_gateway
        if not (multigateway in self.multigateways):
            self.multigateways.add(multigateway)

            self.add_xor_gateway(xor_merge_gateway, *args)
            self.add_and_gateway(and_split_gateway, *args)
            super(MyGraph, self).add_edge(xor_merge_gateway, and_split_gateway)

            self.connect_sources_to_multigateway(sources, xor_merge_gateway)
            self.connect_gateway_to_targets(and_split_gateway, targets)

        return  multigateway

    def connect_and_to_xor(self, sources, targets, *args):
        xor_split_gateway = 'XORs '+str(sources)+'->'+str(targets)
        and_merge_gateway = 'ANDm '+str(sources)+'->'+str(targets)
        multigateway = and_merge_gateway + xor_split_gateway
        if not (multigateway in self.multigateways):
            self.multigateways.add(multigateway)
            self.add_xor_gateway(xor_split_gateway, *args)
            self.add_and_gateway(and_merge_gateway, *args)
            super(MyGraph, self).add_edge(and_merge_gateway, xor_split_gateway)

            self.connect_sources_to_multigateway(sources, and_merge_gateway)
            self.connect_gateway_to_targets(xor_split_gateway, targets)
        return  multigateway

    def connect_sources_to_multigateway(self, sources, gateway):
        for source in sources:
            super(MyGraph, self).add_edge(source, gateway)

    def connect_gateway_to_targets(self, gateway, targets):
        for target in targets:
            super(MyGraph, self).add_edge(gateway, target)

    def add_short_loop(self, source, target):
        print(super(MyGraph, self).in_edges(source))
        print(super(MyGraph, self).out_edges(source))


def get_direct_succesion(events):
    relations = defaultdict(set)
    for sequence in events:
        for i, event in enumerate(sequence[:-1]):
            relations[event].add(sequence[i + 1])
    return relations

def get_parallel_events(causality, successors):
    parallel = []
    for key, elements in causality.items():
        if len(elements) > 1:
            is_parallel = True
            for element in elements:
                n_s = set({ e for e in elements if e != element })
                if not n_s.issubset(successors[element]):
                    is_parallel = False
                    break
            if is_parallel:
                parallel.append(set(elements))
    return parallel


def draw_graphs(df):
    logs = df['Activity'].to_numpy()
    if len(logs) == 0:
        print("No logs in file")
        return

    G = MyGraph()
    start_set_events = {log[0] for log in logs}
    end_set_events = {log[-1] for log in logs}

    direct_successions = get_direct_succesion(logs)
    causality = get_causality(direct_successions)

    inv_causality = get_inv_causality(causality)
    parallel_events = get_parallel_events(causality, direct_successions)

    # adding start event
    G.add_event("start")
    if len(start_set_events) > 1:
        for events in parallel_events:
            if start_set_events <= events:
                G.add_and_split_gateway("start", start_set_events)
        else:
            G.add_xor_split_gateway("start", start_set_events)
    else:
        G.add_edge("start", list(start_set_events)[0])

    # adding split gateways based on causality
    for event in causality:
        if len(causality[event]) > 1:
            for element in causality[event]:
                if (element in inv_causality
                        and len(inv_causality[element]) > 1):
                    break
            else:
                if set(causality[event]) in parallel_events:
                    G.add_and_split_gateway(event, causality[event])
                else:
                    G.add_xor_split_gateway(event, causality[event])
        elif len(causality[event]) == 1:
            target = list(causality[event])[0]
            if event not in end_set_events:
                if target not in inv_causality or len(
                        inv_causality[target]) == 1:
                    G.add_edge(event, target)

    # adding merge gateways based on inverted causality
    for event in inv_causality:
        if len(inv_causality[event]) > 1:
            targets = []
            for element in inv_causality[event]:
                if (element in causality and len(causality[element]) > 1):
                    targets = causality[element]
                    break

            if (len(targets) == 0):
                if set(inv_causality[event]) in parallel_events:
                    G.add_and_merge_gateway(inv_causality[event], event)
                else:
                    G.add_xor_merge_gateway(inv_causality[event], event)
            else:
                if set(inv_causality[event]) in parallel_events:
                    G.connect_and_to_xor(inv_causality[event], targets)
                else:
                    G.connect_xor_to_and(inv_causality[event], targets)

    G.add_end_event("end")
    final_end_gateways = set([
        G.add_xor_split_gateway(event, causality[event])
        for event in end_set_events if event in causality
    ])
    final_end_events = set(
        [event for event in end_set_events if event not in causality])

    end_objects = set(final_end_events.union(final_end_gateways))

    if len(end_objects) > 1:
        for event in parallel_events:
            if set(end_set_events) <= set(event):
                G.add_and_merge_gateway(end_objects, "end")
                break
        else:
            G.add_xor_merge_gateway(end_objects, "end")
    else:
        G.add_edge(list(end_objects)[0], "end")

    G.draw('simple_process_model.png', prog='dot')
    image = Image.open('simple_process_model.png')
    return image



def read_file(file_uploader, separator=',', id="Case ID"):
    if file_uploader.name.endswith(".csv"):
        stringio = StringIO(file_uploader.getvalue().decode("utf-8"))
        df = pd.read_csv(stringio, sep=separator)
    elif file_uploader.name.endswith(".xes"):
        tp = NamedTemporaryFile(delete=False)
        tp.write(file_uploader.getvalue())
        tp.close()
        df = pm4py.convert_to_dataframe(pm4py.read_xes(tp.name))
        os.unlink(tp.name)
        df = df.drop(
            columns=['concept:name', 'lifecycle:transition', 'case:concept:name', 'case:variant', 'case:creator'])
        df.rename(columns={'case:variant-index': id, 'Activity': 'Activity', 'time:timestamp': 'Start Timestamp'}, inplace=True)
    else:
        raise Exception("Invalid file")

    df['Start Timestamp'] = pd.to_datetime(df['Start Timestamp'])
    df = (df
          .sort_values(by=[id, 'Start Timestamp'])
          .groupby([id])
          .agg({'Activity': lambda x: list(x)})
          )

    event_filter = count_and_filter_number_of_occurrences_of_activity(df)

    df['Activity'] = df['Activity'].apply(lambda event_log: list(filter(lambda x: event_filter[x] > 0, event_log)))
    df['Activity'] = df['Activity'].apply(lambda event_log: list(map(lambda x: x + " - " + str(event_filter[x]), event_log)))
    df['Activity'] = df['Activity'].apply(lambda event_log: None if len(event_log) == 0 else event_log)
    df = df[df['Activity'].notna()]
    return df

def count_and_filter_number_of_occurrences_of_activity(df):
    flatten_list = [j for row in df['Activity'] for j in row]
    event_count = Counter(flatten_list)
    max_val = event_count.most_common()[0][1]
    event_tresh = st.slider("Set event treshold", 0, max_val, 0)
    filter_values = [d for d in flatten_list if event_count[d] >= event_tresh]
    counted_filtered_values = Counter(filter_values)
    return counted_filtered_values
