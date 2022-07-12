# Innoviz Demo
This repo was forked from https://github.com/open-mmlab/OpenPCDet. 
[INSTALL](INSTALL.md) flow and [Demo](DEMO.md) was not changed from original repo.

Here we provide a quick demo to run innoviz dataset with dummy model. 
Dummy model - generating dummy detections by adding noise to GT data.

Preliminary requirements:
- Follow installation instructions at [INSTALL](INSTALL.md)
- Download dataset as instructed at [GETTING_STARTED](GETTING_STARTED.md#innoviz-dataset).

to run demo with visuzliation. 
``` 
cd tools
sh scripts/innoviz_demo.sh
```

to run test flow.
``` 
cd tools
sh scripts/innoviz_test.sh
```
