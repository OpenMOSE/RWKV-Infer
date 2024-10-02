# RWKV-Infer
<p align='center'>
<image src="kotori.webp" width=20%/>
    
</p>

<div align="center"> 
A lightweight RWKV inference platform that operates in Cuda and Rocm environments, supporting multi-batch inference.
</div>

## Key Features

- **Multi Recurrent State Sampling**: 

  MRSS (Multiple Recurrent State Sampling) is a novel method for LLM inference that combines multiple fine-tuned states with fixed gating weights to achieve more flexible and effective inference.
   - Pseudo Mixture of State Experts:
By combining multiple states, MRSS integrates knowledge from different "experts," generating richer outputs.

   - Separation of elements: Allows fine-tuning of knowledge, emotions, and speaking styles independently.

   - State reusability: Enables efficient creation of new models through state recombination.

- **Quantization Support**:
  - Int8 (only CUDA)
  - Bitsandbytes NF4 (currently slow)
- **Multi Batch Generation**:
  - True multi batch generation with Flash-Linear-Attention
  - multi batch sampling
  - On an RTX4090, a 7B parameter model can run over 256 batches of inference.

---

> Accelerate your RWKV model inference with RWKV-Infer!


## How To Use
   - 1. Install Latest? Pytorch with Cuda(2.2+ tested)
   - 2. install requirements with triton==2.2.0+(in rocm >=3.0.0)
```sh
pip install -r requirements_fla.txt
```    
   - 3. prepare models in models folder
   - 4. prepare states in states folder
   - 5. Run Server 
```sh
python rwkv_server_fla_fastapi.py --localhost 0.0.0.0 --port 9000 --debug False --workers 64 --dynamic_state_cache_size 512
```     
   - 6. Load Model if quant, set model_strategy:quant
```sh
curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"models/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth","model_viewname":"RWKV x060 1B6 Base","model_strategy":""}'
```
   - 7. Enjoy Infernce via OpenAI Compatible API!


## API Examples
   - 1. Model Load
```sh
curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"models/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth","model_viewname":"RWKV x060 1B6 Base","model_strategy":"","default_temperature":"1.0", "default_top_p":"0.3"}'
```
   - 2. Add Single state
```sh
curl http://127.0.0.1:9000/loadstatemodel -X POST -H "Content-Type: application/json" -d '{"state_filename":"state.pth","state_viewname":"State Test","default_temperature":"1.0", "default_top_p":"0.3"}'
```
   - 3. Add MRSS states
```sh
curl http://127.0.0.1:9000/mrss_loadstatemodel -X POST -H "Content-Type: application/json" -d '{"state_viewname":"MRSS Test", "state_filenames":["states/jp7b-bancho.pth","states/ojousama2.pth","states/secret.pth"], "contain_originalstate":"True", "state_gatingweight":["0.01","0.3","0.4","0.03"],"default_temperature":"1.0", "default_top_p":"0.8"}'
```
   - 4. Remove All State
```sh
curl http://127.0.0.1:9000/removestatemodel -X POST -H "Content-Type: application/json" -d '{"dummy":"dummy"}'
```
   - 5. Get Model Names (During inference, setting the same name as this ID will enable dynamic state loading.)
```sh
curl http://127.0.0.1:9000/models -X GET
```

## Thanks for
   - RWKV-LM,ChatRWKV @BlinkDL
   - rwkv.hpp @harrisonvanderbyl
   - RWKV-PEFT @Jl-er
   - flash-linear-attention @ sustcsonglin


## ToDo for me
   - Improve FLA Stability on bf16 - maybe done.
   - Implement Multi Recurrent State Sampling - done.
   
2024 OpenMOSE
