"""Private namespaces for pinned packages and pure author algorithm symbols."""
from __future__ import annotations

import ast
import importlib
import importlib.abc
import importlib.util
import sys
import types
from pathlib import Path

from .source import BASE, verify_sources


FILE_PATCHES = {
    ("maat", "src/model/detection.py"): [
        ("from xgboost import XGBClassifier", ""),
        ('if feat_filter == "xgboost":', 'if feat_filter == "xgboost":\n            from xgboost import XGBClassifier'),
    ],
    ("tvdiag", "core/model/Encoder.py"): [
        ("from core.model.backbone.tag import TAGEncoder", ""),
        ("from core.model.backbone.SGC import SGCEncoder", ""),
    ],
    ("tvdiag", "core/loss/UnsupervisedContrastiveLoss.py"): [
        ("torch.arange(0, n, dtype=torch.long).cuda()", "torch.arange(0, n, dtype=torch.long, device=z_i.device)"),
    ],
    ("tvdiag", "core/loss/SupervisedContrastiveLoss.py"): [
        ("torch.div(sim, sim_sum)", "torch.div(sim, sim_sum.clamp_min(1e-12))"),
    ],
    ("hades", "hades/codes/models/kpi_model.py"): [
        ("self.t_encoders = [TemporalEncoder(device, input_size=var, **kwargs) for var in self.var_nums]",
         "self.t_encoders = nn.ModuleList([TemporalEncoder(device, input_size=var, **kwargs) for var in self.var_nums])"),
        ("self.t_encoders = [TemporalEncoder(device, 1, **kwargs) for _ in range(self.metric_num)]",
         "self.t_encoders = nn.ModuleList([TemporalEncoder(device, 1, **kwargs) for _ in range(self.metric_num)])"),
    ],
    ("hades", "hades/codes/models/utils.py"): [
        (".squeeze()", ".squeeze(1)"),
    ],
    ("anofusion", "model/GATGRU.py"): [
        ("GATNet(in_c=20, hid_c=self.out_dim, out_c=20, n_heads=6)",
         "GATNet(in_c=self.num_f, hid_c=self.out_dim, out_c=self.num_f, n_heads=6)"),
    ],
    ("anofusion", "model/GAT.py"): [
        ("graph.unsqueeze(0)", "graph"),
        ("outputs.data.masked_fill_(torch.eq(outputs, 0), -float(1e16))",
         "outputs = outputs.masked_fill(graph == 0, -1e16)"),
        ("data[\"graph\"][0]", "data[\"graph\"]"),
    ],
    ("anofusion", "model/GTlayer.py"): [
        ("self.weight = nn.Parameter(torch.Tensor(1, out_channels, in_channels, 1, 1))",
         "self.weight = nn.Parameter(torch.empty(1, out_channels, in_channels, 1, 1)); nn.init.constant_(self.weight, 1.0 / in_channels)"),
    ],
}


class PrivateImports(ast.NodeTransformer):
    def __init__(self, namespace, roots):
        self.namespace, self.roots = namespace, set(roots)

    def visit_ImportFrom(self, node):
        if not node.level and node.module and node.module.split(".")[0] in self.roots:
            node.module = self.namespace + "." + node.module
        return node

    def visit_Import(self, node):
        for name in node.names:
            if name.name.split(".")[0] in self.roots:
                name.asname = name.asname or name.name.split(".")[0]
                name.name = self.namespace + "." + name.name
        return node


class AuthorFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, repo, subdir, roots):
        self.repo, self.roots = repo, roots
        self.root = BASE / "upstream" / repo / subdir
        self.namespace = "_nr_author_" + repo

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self.namespace + "."):
            return None
        relative = fullname[len(self.namespace)+1:].replace(".", "/")
        folder = self.root / relative
        if folder.is_dir():
            return importlib.util.spec_from_loader(fullname, self, is_package=True)
        if folder.with_suffix(".py").is_file():
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        relative = module.__name__[len(self.namespace)+1:].replace(".", "/")
        path = self.root / relative
        if path.is_dir():
            module.__path__ = [str(path)]
            path = path / "__init__.py"
            if not path.is_file():
                return
        else:
            path = path.with_suffix(".py")
        module.__file__ = str(path)
        text = path.read_text()
        relative_file = str(path.relative_to(BASE / "upstream" / self.repo))
        for before, after in FILE_PATCHES.get((self.repo, relative_file), []):
            count = text.count(before)
            if count != 1:
                raise RuntimeError(f"Patch requires one match: {self.repo}/{relative_file}: {before} ({count})")
            text = text.replace(before, after)
        tree = ast.parse(text)
        tree = PrivateImports(self.namespace, self.roots).visit(tree)
        exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), module.__dict__)


def author_package(repo, subdir, module, roots=()):
    verify_sources(repo)
    namespace = "_nr_author_" + repo
    if namespace not in sys.modules:
        finder = AuthorFinder(repo, subdir, roots)
        pkg = types.ModuleType(namespace)
        pkg.__path__ = [str(finder.root)]
        sys.modules[namespace] = pkg
        sys.meta_path.insert(0, finder)
    return importlib.import_module(namespace + "." + module)


def author_symbols(repo, relative, names, bindings):
    """Compile original pure functions/classes without database/CLI side effects."""
    verify_sources(repo)
    path = BASE / "upstream" / repo / relative
    body = [node for node in ast.parse(path.read_text()).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in body} != set(names):
        raise RuntimeError(f"Author symbols not found: {repo}/{relative}")
    namespace = dict(bindings)
    namespace["__name__"] = "_nr_symbols_" + repo
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return types.SimpleNamespace(**{name: namespace[name] for name in names})
