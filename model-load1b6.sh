#!/bin/bash

curl http://127.0.0.1:8000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"models/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth","model_viewname":"RWKV x060 1.6B Base","model_strategy":"cuda fp16i8"}'
