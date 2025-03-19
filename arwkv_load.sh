curl http://127.0.0.1:9000/healthcheck

curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"/home/client/Projects/RWKV-LM-RLHF/main/myfolder/models/ARWKV-7B-CJE-30%.pth","model_viewname":"ARWKV7-Tuned","model_strategy":"fp6","adapter_filename":"","adapter_mode":"bone","endtoken":"<|im_end|>"}'

curl http://127.0.0.1:9000/loadstatemodel -X POST -H "Content-Type: application/json" -d '{"state_filename":"","state_viewname":"State","default_temperature":"1.0", "default_top_p":"0.5"}'