curl http://127.0.0.1:9000/healthcheck

curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"models/PRWKV-7-Phi-4-Instruct-Preview-v0.1.pth","model_viewname":"ARWKV7-Tuned","model_strategy":"fp6","adapter_filename":"/home/client/Projects/NTTProject/arwkv_nsha/rwkv-39.pth","adapter_mode":"", "template":"phi4"}'
