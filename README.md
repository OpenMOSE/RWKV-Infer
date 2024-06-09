# RWKV-Infer

A large-scale RWKV v6 inference engine using the Cuda backend. Supports multi-batch generation and dynamic State switching.

This project aims to simplify the deployment of RWKV model inference in a Docker

## The following features are included:
   - Support for multi-batch generation and stream delivery
   - State switching for each batch
   - OpenAI-compatible API

## How To Use
   - 1. Install Latest Pytorch with Cuda(2.2+ tested)
   - 2. install requirements
```sh
pip install -r requirements.txt
```    
   - 3. prepare models in models folder
   - 4. prepare states in states folder
   - 5. Run Server
```sh
python rwkv_server.py --localhost 0.0.0.0 --port 8000 --debug False --workers 16
```     
   - 6 Load Model
```sh
curl http://127.0.0.1:8000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"models/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth","model_viewname":"RWKV x060 1B6 Base","model_strategy":"cuda fp16"}'
```
   - 7 Enjoy Infernce via OpenAI Compatible API!

## ToDo for me
   - Dynamic State Cache for faster inference
   - Dynamic Swap LoRA(Torch Compile.......)
   - RAG(Cold RAG)
   - Research 4bit inference with 4bit matmul