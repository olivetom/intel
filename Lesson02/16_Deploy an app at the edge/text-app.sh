#:/bin/sh
python ./app.py -i "images/sign.jpg" -t "TEXT" -m "/home/workspace/models/text-detection-0004.xml" -c "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
