#!/bin/bash
curl http://127.0.0.1:9000/healthcheck

 
curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"models/PRWKV7-cxa075-qwen14b-stage2-final.pth","model_viewname":"PRWKV7-cxa075 Qwen 2.5 14B Stage3 16k chunk","model_strategy":"fp6","adapter_filename":"adapters/testlora2.pth","adapter_mode":"lora", "template":"qwen", "endtoken":"<|im_end|>"}'
#
#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"models/ARWKV_7B_R1_16K.pth","model_viewname":"ARWKV-7B-R1 16K","model_strategy":"fp16","adapter_filename":"adapters/arwkv-cje5-6.pth","adapter_mode":"","default_temperature":"1.0", "default_top_p":"0.3"}'

 
 
 
#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"/home/client/Projects/RWKV-LM-RLHF/v6/myfolder/Outputs/14b-sft-bone/rwkv-18-merged.pth","model_viewname":"RWKV x060 14B infctx sft bone r18","model_strategy":"fp8"}'
#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"/home/client/Projects/RWKV-LM-RLHF/v6/myfolder/models/rwkv-x060-7b-world-v3-80_trained-20241025-ctx4k.pth","model_viewname":"RWKV x060 7B Worldv3 80","model_strategy":"fp8"}'
#curl http://127.0.0.1:9000/loadstatemodel -X POST -H "Content-Type: application/json" -d '{"state_filename":"states/ansin_state.pth","state_viewname":"masayo","default_temperature":"1.0", "default_top_p":"0.4"}'


#curl http://127.0.0.1:9000/mrss_loadstatemodel -X POST -H "Content-Type: application/json" -d '{"state_viewname":"Aoko", "state_filenames":["states/jp7b-bancho.pth","states/ojousama2.pth","states/secret.pth"], "contain_originalstate":"True", "state_gatingweight":["0.01","0.3","0.0","0.05"],"default_temperature":"1.0", "default_top_p":"0.6"}'
 