# RWKV-Infer
<p align='center'>
<image src="kotori.webp" width=20%/>
    
</p>

<div align="center"> 
A lightweight RWKV inference platform that operates in Cuda and Rocm environments, supporting multi-batch inference.(RWKV v6,v7)
</div>

## Key Features

- **Multi Recurrent State Sampling**: 

  MRSS (Multi Recurrent State Sampling) is a novel method for LLM inference that combines multiple fine-tuned states with fixed gating weights to achieve more flexible and effective inference.
   - Pseudo Mixture of State Experts:
By combining multiple states, MRSS integrates knowledge from different "experts," generating richer outputs.

   - Separation of elements: Allows fine-tuning of knowledge, emotions, and speaking styles independently.

   - State reusability: Enables efficient creation of new models through state recombination.

- **Mixture of LoRA Experts**:

  Combines multiple LoRA (Low-Rank Adaptation) modules as "experts" that specialize in different tasks or domains 
   - perform inference with the MoLE model trained on RWKV-LM-RLHF.
   - This is a preliminary verification towards the upcoming MoE.

- **Hot swapping of adapter models**: 
  - Bone(Block Affine Transformation) Adapter
  - DoRA: Weight-Decomposed Low-Rank Adaptation
  - LoRA Adapter

- **Quantization Support**:
  - FP8 (Experiment. need NVIDIA H100 or Ada series gpu)
  - FP6 (Early Experiment. slightly degradation. toachao fpx e3m2)
  - FP5 (Early Experiment. ppl 10% degradation. toachao fpx e2m2)
- **Multi Batch Generation**:
  - True multi batch generation with Flash-Linear-Attention(x060, x070)
  - multi batch sampling
  - On an RTX4090, a 7B parameter model can run over 256 batches of inference.
- **Inference Speed examples(RTX4090)**:
<div align=center>

| Batch | 7b_fp8(t/s) | 7b_bf16(t/s)|
|-------|--------|---------|
| 1     | 75     | 50.08   |
| 2     | 132.93 | 90.26   |
| 4     | 284.12 | 192.24  |
| 8     | 545.31 | 368.41  |
| 16    | 967.88 | 731.11  |
| 32    | 1886.72| 1281.16 |
| 64    | 3261.67| 2233.9  |
| 128   | 4792.06| 3629.7  |
| 256   | 5911.39| 4751.32 |
</div>

---

> Accelerate your RWKV model inference with RWKV-Infer!


## How To Use
   - 0. Python >= 3.12
   - 1. Install Latest? Pytorch with Cuda(2.5+ tested) 
   - 2. some case need (conda install libstdcxx -c conda-forge --override-channels) for building cuda kernel
   - 3. install requirements with triton==3.0.0+(in rocm >=3.0.0)
```sh
pip install -r requirements_fla.txt
```    
   - 3. prepare models in models folder
   - 4. prepare states in states folder
   - 5. Run Server 
```sh
python rwkv_server_fla_fastapi.py --localhost 0.0.0.0 --port 9000 --debug False --workers 64 --dynamic_state_cache_size 512
```     
   - 6. Load Model if quant, set model_strategy:bf16,fp16,int8,fp8,nf4
```sh
curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"models/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth","model_viewname":"RWKV x060 1B6 Base","model_strategy":""}'
```
   - 7. Enjoy Infernce via OpenAI Compatible API!


## API Examples
   - 1. Model Load
```sh
curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"models/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth","model_viewname":"RWKV x060 1B6 Base","model_strategy":"","default_temperature":"1.0", "default_top_p":"0.3", "endtoken":"\\n\\n"}'
```
   - 1. Model Load + Adapter
```sh
curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"models/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth","model_viewname":"RWKV x060 1B6 Base","model_strategy":"","adapter_filename":"adapters/rwkv-9-bone.pth","adapter_mode":"bone","default_temperature":"1.0", "default_top_p":"0.3", "endtoken":"\\n\\n"}'
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
