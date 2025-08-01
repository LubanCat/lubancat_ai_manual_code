from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model-dir", help="Path to model directory cloned from HF Hub", default='../')
ap.add_argument("-o", "--output-file", help="Path to save calibration file", default='./data_quant.json')
args = ap.parse_args()

if __name__ == '__main__':
    if not torch.cuda.is_available():
        dev = 'cpu'
    else:            
        dev = 'cuda'

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, trust_remote_code=True)
    model = model.to(dev)
    model = model.eval()
    
    ## 请根据模型特点与使用场景准备用于量化的校准样本。
    input_text = [
        "把这句话翻译成英文: RK3588是新一代高端处理器，具有高算力、低功耗、超强多媒体、丰富数据接口等特点",
        "把这句话翻译成中文: Knowledge can be acquired from many sources. These include books, teachers and practical experience, and each has its own advantages. The knowledge we gain from books and formal education enables us to learn about things that we have no opportunity to experience in daily life. We can also develop our analytical skills and learn how to view and interpret the world around us in different ways."
    ]
    
    calidata = []

    for idx, inp in enumerate(input_text):
        question = inp.strip()
        messages = f"{question}"
        try:
            response, _ = model.chat(tokenizer, messages, history=None, system="You are a translation expert")
            print(response)
            calidata.append({"input":messages, "target":response})
        except:
            calidata.append({"input":messages, "target":''})

    with open(args.output_file, 'w') as f:
        json.dump(calidata, f)