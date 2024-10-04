# CampaignDiagram package


## Install

Old way:
```
python setup.py install --user --single-version-externally-managed --record installed_files.txt

```

Note `installed_files.txt` will list the installed files

New way:

```
python -m build
python -m installer dist/*.whl
```



## Uninstall

Old way:
```
xargs rm -rf <installed_files.txt
```

New way:
```
rm -rf ~/.local/lib/python3.X/site-packages/campaigndiagram*
rm -rf ~/.local/lib/python3.X/site-packages/CampaignDiagram-0.1.dist-info


```


## TODO

Don't install using setup.py directly use pypa/build and/or pypa/install

Use toml file rather than setup.py
