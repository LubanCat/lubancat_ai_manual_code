import numpy as np
import platform
import time
from rknnlite.api import RKNNLite
from tokenization import BertTokenizerForMask

# decice tree for RK356x/RK3576/RK3588
DEVICE_COMPATIBLE_NODE = '/proc/device-tree/compatible'

def get_host():
    # get platform and device type
    system = platform.system()
    machine = platform.machine()
    os_machine = system + '-' + machine
    if os_machine == 'Linux-aarch64':
        try:
            with open(DEVICE_COMPATIBLE_NODE) as f:
                device_compatible_str = f.read()
                if 'rk3562' in device_compatible_str:
                    host = 'RK3562'
                elif 'rk3576' in device_compatible_str:
                    host = 'RK3576'
                elif 'rk3588' in device_compatible_str:
                    host = 'RK3588'
                else:
                    host = 'RK3566_RK3568'
        except IOError:
            print('Read device node {} failed.'.format(DEVICE_COMPATIBLE_NODE))
            exit(-1)
    else:
        host = os_machine
    return host

# model path
RK3566_RK3568_RKNN_MODEL = 'bert-base-uncased_for_rk3566_rk3568.rknn'
RK3588_RKNN_MODEL = 'bert-base-uncased.rknn'
RK3562_RKNN_MODEL = 'bert-base-uncased_for_rk3562.rknn'
RK3576_RKNN_MODEL = 'bert-base-uncased_for_rk3576.rknn'

def postprocess(tokenizer, model_outputs, input_ids, top_k=5):

    # Find masked indices
    model_outputs = np.array(model_outputs)

    masked_index = [i for i, id in enumerate(input_ids[0]) if id == tokenizer.never_split_tokens['[MASK]']]

    # Get logits for masked positions
    logits = model_outputs[0, 0, masked_index, :]

    # Calculate softmax probabilities
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    probs = softmax(logits)

    # Get top k values and indices
    sorted_indices = np.argsort(-probs, axis=-1)
    top_k_indices = sorted_indices[:, :top_k]
    values = probs[np.arange(probs.shape[0])[:, None], top_k_indices]

    result = []
    single_mask = len(masked_index) == 1
    for i in range(len(masked_index)):
        row = []
        for j in range(top_k):
            p = top_k_indices[i, j]
            tokens = input_ids.copy()
            tokens[0][masked_index[i]] = p
            # Filter padding out
            tokens = tokens[tokens != 0]
            sequence = tokenizer.convert_ids_to_tokens(tokens,True)
            proposition = {"score": float(values[i, j]), "token": int(p), "token_str":
                           str(tokenizer.convert_ids_to_tokens([p])), "sequence": str(sequence)}
            row.append(proposition)
        result.append(row)
    if single_mask:
        return result[0]
    return result


if __name__ == '__main__':

    # Get device information
    host_name = get_host()
    if host_name == 'RK3566_RK3568':
        rknn_model = RK3566_RK3568_RKNN_MODEL
    elif host_name == 'RK3562':
        rknn_model = RK3562_RKNN_MODEL
    elif host_name == 'RK3576':
        rknn_model = RK3576_RKNN_MODEL
    elif host_name == 'RK3588':
        rknn_model = RK3588_RKNN_MODEL
    else:
        print("This demo cannot run on the current platform: {}".format(host_name))
        exit(-1)

    rknn_lite = RKNNLite()

    tokenizer = BertTokenizerForMask()

    # Load RKNN model
    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn(rknn_model)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    print('done')

    # input text/tokenizer
    inputs = tokenizer.encode("The capital of France is [MASK].", 16)

    # Init runtime environment
    print('--> Init runtime environment')
    # Run on RK356x / RK3576 / RK3588 with Debian OS, do not need specify target.
    if host_name in ['RK3576', 'RK3588']:
        # For RK3576 / RK3588, specify which NPU core the model runs on through the core_mask parameter.
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    else:
        ret = rknn_lite.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn_lite.inference(inputs=[np.array(inputs['input_ids']),np.array(inputs['attention_mask']),np.array(inputs['token_type_ids'])])

    # Show/save the results
    # np.save('./output.npy', outputs)
    result = postprocess(tokenizer, outputs, np.array(inputs['input_ids']), 3)
    print(result)

    rknn_lite.release()

