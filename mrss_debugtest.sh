#!/bin/bash
curl http://127.0.0.1:9000/healthcheck

#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"/home/client/Projects/RWKV-LM-RLHF/main/myfolder/converted/x060-upgraded.pth","model_viewname":"RWKV x070 7B-JPN Upgrade lr1e-4 e25","model_strategy":"fp16","adapter_filename":"/home/client/Projects/RWKV-LM-RLHF/main/myfolder/Outputs/x070-Convertv7-3/rwkv-3.pth","adapter_mode":"bone","endtoken":"\\n\\n\\x17"}'
#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"models/rwkv-23-merged.pth","model_viewname":"RWKV x070 0.4B ORPO","model_strategy":"fp16","endtoken":"\\n\\n"}'

#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"/home/client/Projects/RWKV-LM-RLHF/main/myfolder/models/rwkv-x070-2b9-world-v3-25%trained-20250103-ctx4k.pth","model_viewname":"RWKV x070 PinkBook","model_strategy":"fp16","adapter_filename":"/home/client/Projects/RWKV-LM-RLHF/rwkv-7.pth","adapter_mode":"bone","endtoken":"\\n\\n"}'
#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"/home/client/Projects/RWKV-LM-RLHF/main/myfolder/models/rwkv-x070-1b5-world-v3-60%trained-20250113-ctx4k.pth","model_viewname":"RWKV x070 1B5 Gen2","model_strategy":"fp16","adapter_filename":"/home/client/Projects/RWKV-LM-RLHF/main/myfolder/Outputs/x070-1b5-GeneralJPENCNV3/rwkv-2.pth","adapter_mode":"bone","endtoken":"\\n\\n\\x17"}'
curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"/home/client/Projects/RWKV-LM-RLHF/main/myfolder/Outputs/rwkv-x070-2b9-cje-instruct-1.pth","model_viewname":"RWKV x070 2B9 CJE Instruct-1","model_strategy":"bf16","adapter_filename":"adapters/rwkv7-2b9-40p-gen3.pth","adapter_mode":"","endtoken":"\\n\\n\\x17"}'
#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"/home/client/Projects/RWKV-LM-RLHF/main/myfolder/converted/x060-upgraded.pth","model_viewname":"RWKV x070 7B CJE Instruct-1","model_strategy":"bf16","adapter_filename":"/home/client/Projects/RWKV-LM-RLHF/main/myfolder/Outputs/x070-7b-cje/rwkv-14.pth","adapter_mode":"bone","endtoken":"\\n\\n\\x17"}'

#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"/home/client/Projects/RWKV-LM-RLHF/main/myfolder/models/RWKV-x060-Jpn-7B-20240816-ctx4096.pth","model_viewname":"RWKV x060 7B","model_strategy":"fp8"}'

#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"models/RWKV-x060-Jpn-7B-20240816-ctx4096.pth","model_viewname":"RWKV x060 7B JPN Redbook","model_strategy":"fp8","adapter_filename":"/home/client/Projects/RWKV-LM-RLHF/main/myfolder/Outputs/General-Redbook/rwkv-6.pth","adapter_mode":"bone","endtoken":"\\n\\n\\x17"}'

#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"models/RWKV-x060-Jpn-7B-20240816-ctx4096.pth","model_viewname":"RWKV x060 7B JPN SFT infctx 32k e17","model_strategy":"fp6"}'
#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"models/RWKV-x060-Jpn-7B-20240816-ctx4096.pth","model_viewname":"RWKV x060 7B JPN SFT infctx 32k e17","model_strategy":"fp6","adapter_filename":"adapters/rwkv-17.pth","adapter_mode":"bone","endtoken":"\\n\\n\\x17"}'

#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"models/RWKV-x060-Jpn-7B-20240816-ctx4096.pth","model_viewname":"RWKV x060 7V WebCoT","model_strategy":"fp8","adapter_filename":"/home/client/Projects/RWKV-LM-RLHF/main/myfolder/Outputs/Browser4-CoT/rwkv-22.pth","adapter_mode":"bone","endtoken":"\\n\\n\\x17"}'


#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"/home/client/Projects/RWKV-LM-RLHF/v6/6B-SRD-nsfw2/rwkv-6b-nsfw-10-merged.pth","model_viewname":"RWKV NSFW 6B","model_strategy":"quantbf16"}'

#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"/home/client/Projects/RWKV-LM-RLHF/v6/0B4-Distillation-lgtm/rwkv0b4-lgtm-2.pth","model_viewname":"RWKV x060 0B4 Distilled","model_strategy":"bf16"}'

#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"/home/client/Projects/RWKV-LM-RLHF/v6/Outputs/7B-Novel2/RWKV-x060-7B-Rosebleu.pth","model_viewname":"RWKV x060 7B JPN","model_strategy":"nf4"}'
#
#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"/home/client/Projects/RWKV-LM-RLHF/v6/myfolder/Outputs/3B-RLHF-DPO-Bancho/rwkv-23-merged.pth","model_viewname":"RWKV x060 3B Bancho DPO","model_strategy":"fp6"}'
#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"//home/client/Projects/RWKV-LM-RLHF/v6/Outputs/1b6-Code-bone/rwkv-0-merged.pth","model_viewname":"RWKV x060 1B6 Code","model_strategy":"bf16"}'

#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"/home/client/Projects/RWKV-LM-RLHF/v6/models/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth","model_viewname":"RWKV x060 1B6 Original","model_strategy":"bf16"}'
#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"models/RWKV-x060-7B-Rosebleu.pth","model_viewname":"LGTM_7B","model_strategy":"fp6"}'


#curl http://127.0.0.1:9000/fjdkgmzak9sd/sksf_appkill -X POST -H "Content-Type: application/json" -d '{"key1":"123874139713915425423541","key2":"d46871245412541544408014"}'
#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"models/RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth","model_viewname":"RWKV x060 3B","model_strategy":""}'

#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"models/RWKV-x060-Jpn-14B-20240819-ctx4096.pth","model_viewname":"RWKV x060 14B JPN","model_strategy":"fp5"}'
#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"/home/client/Projects/RWKV-LM-RLHF/v6/myfolder/Outputs/7bjpn-cot/rwkv-40-merged.pth","model_viewname":"RWKV x060 7B JPN CoT","model_strategy":"fp8"}'

#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"/home/client/Projects/RWKV-LM-RLHF/v6/myfolder/Outputs/14b-sft-bone/rwkv-18-merged.pth","model_viewname":"RWKV x060 14B infctx sft bone r18","model_strategy":"fp8"}'
#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"/home/client/Projects/RWKV-LM-RLHF/v6/myfolder/models/rwkv-x060-7b-world-v3-80_trained-20241025-ctx4k.pth","model_viewname":"RWKV x060 7B Worldv3 80","model_strategy":"fp8"}'
#curl http://127.0.0.1:9000/loadstatemodel -X POST -H "Content-Type: application/json" -d '{"state_filename":"states/ansin_state.pth","state_viewname":"masayo","default_temperature":"1.0", "default_top_p":"0.4"}'
#curl http://127.0.0.1:9000/mrss_loadstatemodel -X POST -H "Content-Type: application/json" -d '{"state_viewname":"masayo mrss", "state_filenames":["states/ansin_state.pth"], "contain_originalstate":"True", "state_gatingweight":["0.9","0.1"],"default_temperature":"1.0", "default_top_p":"0.3"}'

#curl http://127.0.0.1:9000/loadstatemodel -X POST -H "Content-Type: application/json" -d '{"state_filename":"/home/client/Projects/RWKV-LM-RLHF/main/myfolder/Outputs/x070-1b5may-state/rwkv-1-state.pth","state_viewname":"may","default_temperature":"1.0", "default_top_p":"0.5"}'
#curl http://127.0.0.1:9000/mrss_loadstatemodel -X POST -H "Content-Type: application/json" -d '{"state_viewname":"Aoko", "state_filenames":["states/jp7b-bancho.pth","states/ojousama2.pth","states/secret.pth"], "contain_originalstate":"True", "state_gatingweight":["0.01","0.3","0.0","0.05"],"default_temperature":"1.0", "default_top_p":"0.6"}'

#curl http://127.0.0.1:9000/mrss_set_gatingweight -X POST -H "Content-Type: application/json" -d '{"state_viewname":"MRSS Test","state_gatingweight":["0.01","0.1","0.9","0.01"],"default_temperature":"1.2", "default_top_p":"0.8"}'

#curl http://127.0.0.1:9000/loadstatemodel -X POST -H "Content-Type: application/json" -d '{"state_filename":"/home/client/Projects/RWKV-LM-RLHF/main/myfolder/Outputs/Browser4-CoT/rwkv-22-state.pth","state_viewname":"State","default_temperature":"1.2", "default_top_p":"0.5"}'
