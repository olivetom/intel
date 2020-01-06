import argparse
### TODO: Load the necessary libraries
from openvino.inference_engine import IENetwork, IECore

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Load an IR into the Inference Engine")
    # -- Create the descriptions for the commands
    m_desc = "The path and filename without extension of the model XML file"

    # -- Create the arguments
    parser.add_argument("-m", help=m_desc)
    args = parser.parse_args()

    return args


def load_to_IE(model_xml):
    ### TODO: Load the Inference Engine API
    plugin = IECore()
#     iecore = IECore()
#     print(iecore.available_devices)
    

    ### TODO: Load IR files into their related class
    model = model_xml+"xml"
    weights = model_xml+"bin"
    net = IENetwork(model=model, weights=weights)

    ### TODO: Add a CPU extension, if applicable. It's suggested to check
    ###       your code for unsupported layers for practice before 
    ###       implementing this. Not all of the models may need it.
    plugin.add_extension("/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so", "CPU")

    
    
        
    ### TODO: Get the supported layers of the network
    ## layers_map = iecore.query_network(network=net, device_name="HETERO:GPU,CPU") fails because libclDNNPlugin.so fails to find GPU
    supported_layers = plugin.query_network(network=net, device_name="CPU")
    
#     for layer in layers_map:
#         print(layer.values())
        
    ### TODO: Check for any unsupported layers, and let the user
    ###       know if anything is missing. Exit the program, if so.

    unsupported_layers = [l for l in net.layers.keys() if l not in supported_layers]
    if len(unsupported_layers) > 0:
        print("Can't run on device CPU because there are unsupported layers.")
        print(unsupported_layers)
        print("exiting...")
        exit(-1)
    
    
    ### TODO: Load the network into the Inference Engine
    exec_net = plugin.load_network(network=net, device_name="CPU", num_requests=1)
   
    print("IR successfully loaded into Inference Engine.")

    return

def main():
    args = get_args()
    load_to_IE(args.m)


if __name__ == "__main__":
    main()
