#!/bin/bash

curl http://127.0.0.1:9000/loadmodel -X POST -H "Content-Type: application/json" -d '{"model_filename":"models/RWKV-x060-Jpn-7B-20240816-ctx4096.pth","model_viewname":"RWKV x060 7B JPN","model_strategy":""}'


curl http://127.0.0.1:9000/mrss_loadstatemodel -X POST -H "Content-Type: application/json" -d '{"state_viewname":"MRSS Test", "state_filenames":["states/jp7b-bancho.pth","states/ojousama2.pth","states/secret.pth"], "contain_originalstate":"True", "state_gatingweight":["0.05","0.5","0.0","0.05"],"default_temperature":"1.0", "default_top_p":"0.8"}'

curl http://127.0.0.1:9000/mrss_set_gatingweight -X POST -H "Content-Type: application/json" -d '{"state_viewname":"MRSS Test","state_gatingweight":["0.00","0.0","0.9","0.05"],"default_temperature":"1.2", "default_top_p":"0.9"}'

curl http://127.0.0.1:9000/loadstatemodel -X POST -H "Content-Type: application/json" -d '{"state_filename":"states/blunt.pth","state_viewname":"Blunt san","default_temperature":"1.0", "default_top_p":"0.3"}'
