#!/bin/bash
curl http://127.0.0.1:9000/healthcheck

#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"models/hxa079-qwen3-8b-stage2-hybrid-final.pth","model_viewname":"hxa079-8B","model_strategy":"nf4","adapter_filename":"","adapter_mode":"", "template":"qwen3", "endtoken":"<|im_end|>","default_temperature":"1.0", "default_top_p":"0.3"}'

curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"/home/client/Projects/RWKV-Reka-3.1-Flash/","model_viewname":"RWKV-Reka May","model_strategy":"int8","adapter_filename":"/home/client/Projects/RWKV-LM-RLHF/main/myfolder/hxa079_output_may/rwkv-3.pth","adapter_mode":"bone", "template":"rekaflash31", "endtoken":"\n <sep>","default_temperature":"0.6", "default_top_p":"0.3"}'
#
#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"models/PRWKV7-cxa075-qwen14b-stage2-final.pth","model_viewname":"PRWKV7-cxa075 Qwen 2.5 14B Stage3 16k e7","model_strategy":"fp6","adapter_filename":"adapters/testlora7.pth","adapter_mode":"lora", "template":"qwen", "endtoken":"<|im_end|>","default_temperature":"0.7", "default_top_p":"0.1"}'
#
#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"models/ARWKV_7B_R1_16K.pth","model_viewname":"ARWKV-7B-R1 16K","model_strategy":"fp16","adapter_filename":"adapters/arwkv-cje5-6.pth","adapter_mode":"","default_temperature":"1.0", "default_top_p":"0.3"}'

 
 
#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"/home/client/Projects/RWKV-LM-RLHF/v6/myfolder/Outputs/14b-sft-bone/rwkv-18-merged.pth","model_viewname":"RWKV x060 14B infctx sft bone r18","model_strategy":"fp8"}'
#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"/home/client/Projects/RWKV-LM-RLHF/v6/myfolder/models/rwkv-x060-7b-world-v3-80_trained-20241025-ctx4k.pth","model_viewname":"RWKV x060 7B Worldv3 80","model_strategy":"fp8"}'
#curl http://127.0.0.1:9000/loadstatemodel -X POST -H "Content-Type: application/json" -d '{"state_filename":"/home/client/Projects/RWKV-LM-RLHF/main/myfolder/Outputs/x070-2b9-pre/rwkv-2-state.pth","state_viewname":"Prefix","default_temperature":"1.0", "default_top_p":"0.4"}'


#curl http://127.0.0.1:9000/mrss_loadstatemodel -X POST -H "Content-Type: application/json" -d '{"state_viewname":"Aoko", "state_filenames":["states/jp7b-bancho.pth","states/ojousama2.pth","states/secret.pth"], "contain_originalstate":"True", "state_gatingweight":["0.01","0.3","0.0","0.05"],"default_temperature":"1.0", "default_top_p":"0.6"}'
 

#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"models/rwkv7-g0-7.2b-20250722-ctx4096.pth","model_viewname":"RWKV x070 DFT lr1e-5 b32 e5 a0.15","model_strategy":"int8","adapter_filename":"adapters/rwkv-5.pth","adapter_mode":"lora", "template":"world", "endtoken":"\n\n","default_temperature":"1.0", "default_top_p":"0.3"}'

#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"models/rwkv7-g0-7.2b-20250722-ctx4096.pth","model_viewname":"RWKV x070 DFT psycho","model_strategy":"int8","adapter_filename":"adapters/psycho.pth","adapter_mode":"lora", "template":"world", "endtoken":"\n\n","default_temperature":"1.0", "default_top_p":"0.3"}'
