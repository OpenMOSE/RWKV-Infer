#!/bin/bash
curl http://127.0.0.1:9000/healthcheck

#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"models/RWKV-x060-Jpn-7B-20240816-ctx4096.pth","model_viewname":"RWKV x060 7B JPN","model_strategy":"fp5"}'


curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"models/RWKV-x060-Jpn-7B-20240816-ctx4096.pth","model_viewname":"RWKV x060 7B JPN SFT infctx 32k e51 fs1.8","model_strategy":"fp8","adapter_filename":"adapters/7b-kure-40-bone.pth","adapter_mode":"bone","endtoken":"\\n\\n\\x17"}'

#curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"models/RWKV-x060-Jpn-14B-20240819-ctx4096.pth","model_viewname":"RWKV x060 14B JPN CoT RC0","model_strategy":"fp8","adapter_filename":"adapters/14b-general-23-bone.pth","adapter_mode":"bone"}'


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
#curl http://127.0.0.1:9000/loadstatemodel -X POST -H "Content-Type: application/json" -d '{"state_filename":"/home/client/Projects/RWKV-LM-State-4bit-Orpo/v6-state-4bit/20241108_14borpo3/rwkv-7.pth","state_viewname":"state-tuning r7 or0.1","default_temperature":"1.0", "default_top_p":"0.3"}'
#curl http://127.0.0.1:9000/loadstatemodel -X POST -H "Content-Type: application/json" -d '{"state_filename":"states/6bjpn-bancho2.pth","state_viewname":"bancho","default_temperature":"1.0", "default_top_p":"0.3"}'
#curl http://127.0.0.1:9000/mrss_loadstatemodel -X POST -H "Content-Type: application/json" -d '{"state_viewname":"Aoko", "state_filenames":["states/jp7b-bancho.pth","states/ojousama2.pth","states/secret.pth"], "contain_originalstate":"True", "state_gatingweight":["0.01","0.3","0.0","0.05"],"default_temperature":"1.0", "default_top_p":"0.6"}'

#curl http://127.0.0.1:9000/mrss_set_gatingweight -X POST -H "Content-Type: application/json" -d '{"state_viewname":"MRSS Test","state_gatingweight":["0.01","0.1","0.9","0.01"],"default_temperature":"1.2", "default_top_p":"0.8"}'

#curl http://127.0.0.1:9000/loadstatemodel -X POST -H "Content-Type: application/json" -d '{"state_filename":"states/blunt.pth","state_viewname":"Blunt","default_temperature":"1.0", "default_top_p":"0.3"}'
