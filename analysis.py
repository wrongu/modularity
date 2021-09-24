import torch
from pathlib import Path
from models import LitWrapper
from itertools import product
from util import merge_dicts
from pandas import DataFrame


def last_model(model_dir):
    return model_dir / 'weights' / 'last.ckpt'


def best_model(model_dir):
    data = torch.load(last_model(model_dir))
    for v in data['callbacks'].values():
        if 'best_model_path' in v:
            return v['best_model_path']
    return None


def gather_metrics(ckpt_file, metrics):
    info = torch.load(ckpt_file, map_location='cpu')
    keyed_rows = {}

    for metric in metrics:
        parts = metric.split('.')
        if parts[0] == "modules":
            # Parse metric of the form "modules.<assoc_method>.<layer_number>.<field>"
            # Create a row for each unique combination of assoc_method and layer_number
            assoc, layer, field = parts[1], int(parts[2]), parts[3]
            # Update this 'field' for the row specified by 'assoc' and 'layer'
            _key = f"{assoc}_{layer}"
            try:
                _data = {"assoc": assoc, "layer": layer, field: info["modules"][assoc][layer][field]}
            except Exception as e:
                if not (isinstance(e, KeyError) or isinstance(e, IndexError)):
                    raise e
                _data = {"assoc": assoc, "layer": layer, field: float('nan')}
        elif parts[0] == "align":
            # Parse metric of the form "aslign.<assoc_method_1>:<assoc_method_2>.<layer_number>.<field>". Create a new
            # row per unique pair of association methods.
            (assoc_a, assoc_b), layer, field = sorted(parts[1].split(':')), int(parts[2]), parts[3]
            # Update this 'field' for the row specified by the two 'assoc' methods (in sorted order) and 'layer'
            _key = f"{assoc_a}_{assoc_b}_{layer}"
            try:
                _data = {"assoc_a": assoc_a, "assoc_b": assoc_b, "layer": layer,
                         field: info["align"][assoc_a + ":" + assoc_b][layer][field]}
            except Exception as e:
                if not (isinstance(e, KeyError) or isinstance(e, IndexError)):
                    raise e
                _data = {"assoc_a": assoc_a, "assoc_b": assoc_b, "layer": layer, field: float('nan')}
        else:
            # No parsing in default case. Just get values and add to existing row.
            _key = "basic"
            try:
                _data = {metric: info[metric]}
            except KeyError:
                _data = {metric: float('nan')}

        # Convert out of torch to play more nicely with pandas
        def cast_type(val):
            if torch.is_tensor(val):
                if val.numel() == 1:
                    return val.item()
                else:
                    return val.numpy()
            else:
                return val
        _data = {k: cast_type(v) for k, v in _data.items()}

        # Insert or update data stored at keyed_rows[_key]
        keyed_rows[_key] = merge_dicts(keyed_rows.get(_key, {}), _data)

    if "basic" in keyed_rows and len(keyed_rows) > 1:
        # Requested some 'basic' stats as well as either 'modules' or 'align'; copy basic info to all other rows
        basic_metrics = keyed_rows.pop("basic")
        for k, v in keyed_rows.items():
            v.update(basic_metrics)

    return keyed_rows.values()


def generate_model_specs(base_specs):
    if isinstance(base_specs, dict):
        base_specs = [base_specs]

    for spec in base_specs:
        iterable_fields = {k: v for k, v in spec.items() if (hasattr(v, '__iter__') and not isinstance(v, str))}
        for vv in product(*iterable_fields.values()):
            yield merge_dicts(spec, dict(zip(iterable_fields.keys(), vv)))


def get_model_checkpoint(spec, log_dir=Path('logs/')):
    model = LitWrapper(**spec)
    which_checkpoint = spec.get('checkpoint', 'last')
    if which_checkpoint == 'last':
        return last_model(log_dir / model.get_uid())
    elif which_checkpoint == 'best':
        return best_model(log_dir / model.get_uid())
    else:
        raise ValueError(f"spec['checkpoint'] must be 'last' or 'best' but as '{which_checkpoint}'")


def load_data_as_table(model_specs, metrics, log_dir=Path('logs/')):
    tbl = DataFrame()
    for spec in model_specs:
        new_rows = gather_metrics(get_model_checkpoint(spec, log_dir), metrics)
        tbl = tbl.append(DataFrame([merge_dicts(spec, r) for r in new_rows]))
    return tbl


if __name__ == "__main__":
    # simple test..
    spec = {'dataset': 'mnist', 'task': 'sup', 'drop': 0., 'l1': 1e-4, 'l2': 1e-5, 'run': 6}
    df = load_data_as_table([spec], ['modules.forward_cov.0.score', 'modules.forward_cov.1.score', 'test_acc'])
