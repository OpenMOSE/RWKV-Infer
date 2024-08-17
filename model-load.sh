#!/bin/bash

curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"models/rwkv-x060-14b-world-v2.1-81%trained-20240527-ctx4k.pth","model_viewname":"RWKV x060 14B Base","model_strategy":"quant"}'
