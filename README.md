# RWKV-Infer with Flash-Linear-Attention

A large-scale RWKV v6 inference engine using the triton backend. Supports true multi-batch generation and dynamic State switching.

This project aims to simplify the deployment of RWKV model inference in a Docker

When inferring quantized models, pre-prompt processing is twice as fast as the conventional RWKV-Infer.

## The following features are included:
   - Support for true multi-batch generation and stream delivery
   - State switching for each batch
   - Bitsandbytes Quantization Support(NF4)
   - OpenAI-compatible API
   - Dynamic RNN State Cache
   By dynamically caching RNN states, we have improved the efficiency of state regeneration frequency and accelerated inference speed.

## How To Use
   - 1. Install Latest? Pytorch with Cuda(2.2+ tested)
   - 2. install requirements with triton==2.2.0
```sh
pip install -r requirements.txt
```    
   - 3. prepare models in models folder
   - 4. prepare states in states folder
   - 5. Run Server 
```sh
python rwkv_server.py --localhost 0.0.0.0 --port 9000 --debug False --workers 16 --dynamic_state_cache_size 64
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

## Thanks for
RWKV-LM,ChatRWKV @BlinkDL
rwkv.hpp @harrisonvanderbyl
RWKV-PEFT @Jl-er
flash-linear-attention @ sustcsonglin


## ToDo for me
   - Improve FLA Stability on bf16
   - Mixture of State Experts 
   
2024 OpenMOSE
