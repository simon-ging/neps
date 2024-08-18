# main changes in fork branch

## multigpu lightning example script

however those are new files so no merge problems.

## problems with string "error" appearing (instead of nan-floats) and breaking everything

https://github.com/simon-ging/neps/commit/4ed0a2f824be478294c18ce69c975f0f0e0670e3

neps/optimizers/multi_fidelity/promotion_policy.py L65

```python
if len(self.rung_members_performance[rung]) > 0:
    self.rung_members_performance[rung][
        self.rung_members_performance[rung] == "error"] = np.nan
    # turn numpy array dtype 'O' back to float
    self.rung_members_performance[rung] = self.rung_members_performance[rung].astype(float)
```

neps/optimizers/multi_fidelity_prior/utils.py L156

```python
REMOVE
if not np.isnan(observed_configs.at[i, "perf"])

ADD
if not observed_configs.at[i, "perf"] == "error"
    and not np.isnan(observed_configs.at[i, "perf"])

```

## missing result files breaking everything

same commit as above

neps/runtime.py L284

```python
REMOVE
result = deserialize(disk.result_file)

ADD
if Path(disk.result_file).is_file():
    result = deserialize(disk.result_file)
else:
   logger.error(f"Error file found but no result file {disk.result_file}", exc_info=True)
    result = {}
```

neps/status/status.py L50

```python
if trial.report is None:
    breakpoint()
    continue
```




```python

```




```python

```


