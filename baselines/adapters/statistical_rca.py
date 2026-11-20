"""Author RCA algorithms over the same strictly historical RE2 windows."""
from __future__ import annotations

import datetime
import importlib
import importlib.util
import math
import sys
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache, reduce
from typing import FrozenSet, List

import numpy as np
import pandas as pd

from baselines.author_imports import author_package, author_symbols
from baselines.source import BASE, verify_sources


def author_causal_patches():
    """Install the author's two runtime overrides in this worker process only."""
    verify_sources("rcd")
    for name, relative in (
        ("causallearn.graph.GraphClass", "causallearn/graph/GraphClass.py"),
        ("causallearn.utils.PCUtils.SkeletonDiscovery", "causallearn/utils/PCUtils/SkeletonDiscovery.py"),
    ):
        parent = importlib.import_module(name.rsplit(".", 1)[0])
        path = BASE / "upstream/rcd" / relative
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        setattr(parent, name.rsplit(".", 1)[1], module)


class RCDAuthor:
    def __init__(self, services, seed=42):
        author_causal_patches()
        self.api = author_package("rcd", "", "rcd", ("utils",))
        self.services, self.seed = services, seed

    def __call__(self, normal, current, **kwargs):
        result = self.api.top_k_rc(normal.copy(), current.copy(), k=len(normal.columns),
                                   bins=5, gamma=5, seed=self.seed, localized=True, verbose=False)
        return ranked_features(result["root_cause"], self.services)


def ranked_features(ranks, services):
    scores = np.zeros(len(services), dtype=float)
    index = {s: i for i, s in enumerate(services)}
    seen = set()
    for position, item in enumerate(ranks):
        if isinstance(item, (tuple, list)):
            item = item[0]
        service = str(item).rsplit("_", 1)[0] if str(item) not in index else str(item)
        if service in index and service not in seen:
            scores[index[service]] = 1.0/(position+1)
            seen.add(service)
    return scores if seen else None


class CIRCAAuthor:
    def __init__(self, services, edges, reference, entry):
        self.services, self.reference, self.entry = services, reference, entry
        self.graph_module = author_package("circa", "", "circa.model.graph")
        common = author_package("circa", "", "circa.alg.common")
        graph_factory = author_package("circa", "", "circa.graph.common")
        ci = author_package("circa", "", "circa.alg.ci")
        self.case = author_package("circa", "", "circa.model.case")
        self.loader = author_package("circa", "", "circa.model.data_loader")
        import networkx as nx
        self.columns = list(reference.columns)
        Node = self.graph_module.Node
        self.nodes = {c: Node(*c.rsplit("_", 1)) for c in self.columns}
        graph = nx.DiGraph()
        graph.add_nodes_from(self.nodes.values())
        for column, node in self.nodes.items():
            service, metric = column.rsplit("_", 1)
            latency = self.nodes.get(service+"_latency-90")
            if latency and node != latency:
                graph.add_edge(node, latency)
        for caller, callee in edges:
            a, b = self.nodes.get(callee+"_latency-90"), self.nodes.get(caller+"_latency-90")
            if a and b:
                graph.add_edge(a, b)
        self.graph = self.graph_module.MemoryGraph(graph)
        self.model = common.Model(graph_factory=graph_factory.StaticGraphFactory(self.graph),
                                  scorers=[ci.RHTScorer()])

    def __call__(self, normal, current, **kwargs):
        # Synthetic times express normal-reference/candidate segments, not fault time.
        # Every value comes from train-normal or the permitted five-minute history.
        joined = pd.concat([normal, current], ignore_index=True)
        values = defaultdict(dict)
        for col, node in self.nodes.items():
            values[node.entity][node.metric] = list(enumerate(joined[col].to_numpy(dtype=float)))
        sli = self.nodes.get(self.entry+"_latency-90")
        if sli is None:
            raise ValueError("CIRCA needs an observable training-defined business SLI")
        case = self.case.CaseData(self.loader.MemoryDataLoader(values), sli=sli,
                                 detect_time=len(joined)-1, interval=datetime.timedelta(seconds=1),
                                 lookup_window=len(joined)-1, detect_window=len(current), prune=False)
        result = self.model.analyze(case, current=len(joined)-1)
        scores = np.zeros(len(self.services))
        index = {s: i for i, s in enumerate(self.services)}
        for node, score in result:
            if node.entity in index:
                scores[index[node.entity]] = max(scores[index[node.entity]], float(score.score))
        return scores


class TraceRCAAuthor:
    def __init__(self, services):
        from loguru import logger
        from mlxtend.frequent_patterns import fpgrowth
        from mlxtend.preprocessing import TransactionEncoder
        # The author only scores singleton patterns in its default non-PRFL path.
        # Limit mining to singletons without changing those singleton supports.
        def singleton_patterns(*args, **kwargs):
            return fpgrowth(*args, **kwargs, max_len=1)
        self.api = author_symbols("tracerca", "run_localization_association_rule_mining_20210516.py",
                                  ["ItemsetHandler", "TraceRCA"],
                                  dict(np=np, pd=pd, defaultdict=defaultdict, lru_cache=lru_cache,
                                       reduce=reduce, FrozenSet=FrozenSet, List=List, logger=logger,
                                       fpgrowth=singleton_patterns, TransactionEncoder=TransactionEncoder))
        self.services = services

    def __call__(self, invocations):
        if invocations.empty or not invocations.predict.any():
            return None
        ranks = self.api.TraceRCA()(invocations.copy(), min_support_rate=0.1,
                                    quiet=True, k=len(self.services), enable_PRFL=False)
        return ranked_features(ranks, self.services)


class MicroRankAuthor:
    def __init__(self, services):
        self.services = services
        self.graph = author_symbols("microrank", "preprocess_data.py", ["get_pagerank_graph"],
                                    dict(deepcopy=deepcopy))
        self.walk = author_symbols("microrank", "pagerank.py",
                                   ["trace_pagerank", "normalization", "firstPr", "pageRank"], dict(np=np))
        self.spectrum = author_symbols("microrank", "online_rca.py",
                                       ["calculate_spectrum_without_delay_list"], dict(np=np, math=math))

    def __call__(self, spans, invocations):
        if invocations.empty:
            return None
        predicted = invocations.groupby("trace_id").predict.max()
        abnormal = list(predicted[predicted].index)
        normal = list(predicted[~predicted].index)
        if not abnormal or not normal:
            return None
        known = {(s["trace"], s["span"]) for s in spans}
        records = [{"_source": {"traceID": s["trace"], "spanID": s["span"],
                                  "operationName": s["operation"],
                                  "references": [{"spanID": s["parent"]}] if (s["trace"], s["parent"]) in known else [],
                                  "process": {"tags": [{"key": "name", "value": s["service"]}]},
                                  "tags": []}} for s in sorted(spans, key=lambda s: (s["trace"], s["start"]))]
        class IndexedTraceList(list):
            def __init__(self, values):
                super().__init__(values)
                self.members = set(values)

            def __contains__(self, item):
                return item in self.members

        # Keep the author's list order and indexed access, speed up membership
        # tests only. This changes no request graph or PageRank calculation.
        a = self.graph.get_pagerank_graph(IndexedTraceList(abnormal), records)
        n = self.graph.get_pagerank_graph(IndexedTraceList(normal), records)
        ar, ac = self.walk.trace_pagerank(*a, anomaly=True)
        nr, nc = self.walk.trace_pagerank(*n, anomaly=False)
        ranks, values = self.spectrum.calculate_spectrum_without_delay_list(
            ar, nr, len(abnormal), len(normal), len(ar)+len(nr), nc, ac, "ochiai")
        scores = np.zeros(len(self.services))
        operation_services = {s["service"]+"_"+s["operation"]: s["service"] for s in spans}
        for name, value in zip(ranks, values):
            service = operation_services.get(name)
            if service in self.services and math.isfinite(float(value)):
                idx = self.services.index(service)
                scores[idx] = max(scores[idx], float(value))
        return scores if np.any(scores) else None


class TORAIAuthor:
    def __init__(self, services):
        author_causal_patches()
        self.api = author_package("rcaeval", "", "RCAEval.e2e.torai", ("RCAEval",))
        self.services = services

    def __call__(self, normal, current, normal_logs, current_logs):
        boundary = len(normal)
        metric = pd.concat([normal, current], ignore_index=True)
        metric.insert(0, "time", np.arange(len(metric)))
        logs = pd.concat([normal_logs, current_logs], ignore_index=True)
        logs.insert(0, "time", np.arange(len(logs)))
        result = self.api.torai({"metric": metric, "logts": logs}, inject_time=boundary,
                                dataset=None, localized=True, bins=5, verbose=False)
        return ranked_features(result["ranks"], self.services)
