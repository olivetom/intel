#:/bin/sh
python ./app.py -i "images/sitting-on-car.jpg" -t "POSE" -m "/home/workspace/models/human-pose-estimation-0001.xml" -c "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
