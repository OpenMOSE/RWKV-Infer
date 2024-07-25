# RWKV-Infer

A large-scale RWKV v6 inference engine using the Cuda backend. Supports multi-batch generation and dynamic State switching.

This project aims to simplify the deployment of RWKV model inference in a Docker
## 2024.07.25 Update:
   - Support for CPU memory storage of Dynamic State Cache
   - Improved behavior during inference interruption
   - Speedup for 8-bit inference (5-13% on AD102)
   - FastAPI implementation
   
   Since State Cache can be stored virtually infinitely, re-inference can be avoided in most cases

## The following features are included:
   - Support for multi-batch generation and stream delivery
   - State switching for each batch
   - OpenAI-compatible API
   - Dynamic RNN State Cache(20240610)
   
   By dynamically caching RNN states, we have improved the efficiency of state regeneration frequency and accelerated inference speed.

## How To Use
   - 1. Install Latest Pytorch with Cuda(2.2+, 2.4Tested)
   - 2. install requirements
```sh
pip install -r requirements.txt
```    
   - 3. prepare models in models folder
   - 4. prepare states in states folder
   - 5. Run Server 
```sh
python rwkv_server.py --localhost 0.0.0.0 --port 8000 --debug False --workers 16 --dynamic_state_cache_size 64
```     
   - 6. Load Model
```sh
curl http://127.0.0.1:8000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"models/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth","model_viewname":"RWKV x060 1B6 Base","model_strategy":"cuda fp16"}'
```
   - 7. Enjoy Infernce via OpenAI Compatible API!


## API Examples
   - 1. Model Load
```sh
curl http://127.0.0.1:8000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"models/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth","model_viewname":"RWKV x060 1B6 Base","model_strategy":"cuda fp16"}'
```
   - 2. Add State
```sh
curl http://127.0.0.1:8000/loadstatemodel -X POST -H "Content-Type: application/json" -d '{"state_filename":"state.pth","state_viewname":"State Test"}'
```
   - 3. Remove All State
```sh
curl http://127.0.0.1:8000/removestatemodel -X POST -H "Content-Type: application/json" -d '{"dummy":"dummy"}'
```
   - 4. Get Model Names (During inference, setting the same name as this ID will enable dynamic state loading.)
```sh
curl http://127.0.0.1:8000/models -X GET
```


## ToDo for me
   - (Done)Dynamic State Cache for faster inference 
   - Dynamic Swap LoRA(Torch Compile.......)
   - RAG(Cold RAG)
   - Research 4bit inference with 4bit matmul
