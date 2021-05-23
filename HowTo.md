#### Plotting results

> https://spinningup.openai.com/en/latest/user/plotting.html

```
python -m spinup.run plot [path/to/output_directory ...] [--legend [LEGEND ...]]
    [--xaxis XAXIS] [--value [VALUE ...]] [--count] [--smooth S]
    [--select [SEL ...]] [--exclude [EXC ...]]
```

#### Running trained policy

> https://spinningup.openai.com/en/latest/user/saving_and_loading.html#loading-and-running-trained-policies

```
python -m spinup.run test_policy path/to/output_directory
```