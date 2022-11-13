import asyncio
import json
from transformers import RobertaTokenizer, T5ForConditionalGeneration, T5Config
import torch
import onnxruntime
import numpy as np
import pickle
from fastapi import FastAPI, Request

app = FastAPI()

def main(code: list, gpu: bool = False) -> dict:
    """Generate vulnerability predictions and line scores.
    Parameters
    ----------
    code : :obj:`list`
        A list of String functions.
    gpu : bool
        Defines if CUDA inference is enabled
    Returns
    -------
    :obj:`dict`
        A dictionary with two keys, "batch_vul_pred", "batch_vul_pred_prob", and "batch_line_scores"
        "batch_vul_pred" stores a list of vulnerability prediction: [0, 1, ...] where 0 means non-vulnerable and 1 means vulnerable
        "batch_vul_pred_prob" stores a list of vulnerability prediction probabilities [0.89, 0.75, ...] corresponding to "batch_vul_pred"
        "batch_line_scores" stores line scores as a 2D list [[att_score_0, att_score_1, ..., att_score_n], ...]
    """
    provider = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
    # load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("./inference-common/tokenizer")
    model_input = tokenizer(code, truncation=True, max_length=512, padding='max_length',
                            return_tensors="pt").input_ids
    # onnx runtime session
    ort_session = onnxruntime.InferenceSession("./saved_models/onnx_checkpoint/linevul.onnx", providers=provider)
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(model_input)}
    prob, attentions = ort_session.run(None, ort_inputs)
    # prepare token for attention line score mapping
    batch_tokens = []
    for mini_batch in model_input.tolist():
        tokens = tokenizer.convert_ids_to_tokens(mini_batch)
        tokens = [token.replace("Ġ", "") for token in tokens]
        tokens = [token.replace("ĉ", "Ċ") for token in tokens]
        batch_tokens.append(tokens)
    batch_att_weight_sum = []
    # access each layer
    for j in range(len(attentions)):
        att_weight_sum = None
        att_of_one_func = attentions[j]
        for i in range(len(attentions[0])):
            layer_attention = att_of_one_func[i]
            # summerize the values of each token dot other tokens
            layer_attention = sum(layer_attention)
            if att_weight_sum is None:
                att_weight_sum = layer_attention
            else:
                att_weight_sum += layer_attention
        # normalize attention score
        att_weight_sum -= att_weight_sum.min()
        att_weight_sum /= att_weight_sum.max()
        batch_att_weight_sum.append(att_weight_sum)
    # batch_line_scores (2D list with shape of [batch size, seq length]): [[att_score_0, att_score_1, ..., att_score_n], ...]
    batch_line_scores = []
    for i in range(len(batch_att_weight_sum)):
        # clean att score for <s> and </s>
        att_weight_sum = clean_special_token_values(batch_att_weight_sum[i], padding=True)
        # attention should be 1D tensor with seq length representing each token's attention value
        word_att_scores = get_word_att_scores(tokens=batch_tokens[i], att_scores=att_weight_sum)
        line_scores = get_all_lines_score(word_att_scores)
        batch_line_scores.append(line_scores)
    # batch_vul_pred (1D list with shape of [batch size]): [pred_1, pred_2, ..., pred_n]
    batch_vul_pred = np.argmax(prob, axis=-1).tolist()
    # batch_vul_pred_prob (1D list with shape of [batch_size]): [prob_1, prob_2, ..., prob_n]
    batch_vul_pred_prob = []
    for i in range(len(prob)):
        batch_vul_pred_prob.append(prob[i][batch_vul_pred[i]].item())  # .item() added to prevent 'Object of type float32 is not JSON serializable' error

    return {"batch_vul_pred": batch_vul_pred, "batch_vul_pred_prob": batch_vul_pred_prob,
            "batch_line_scores": batch_line_scores}


def get_word_att_scores(tokens: list, att_scores: list) -> list:
    word_att_scores = []
    for i in range(len(tokens)):
        token, att_score = tokens[i], att_scores[i]
        word_att_scores.append([token, att_score])
    return word_att_scores


def get_all_lines_score(word_att_scores: list):
    # word_att_scores -> [[token, att_value], [token, att_value], ...]
    separator = "Ċ"
    # to return
    all_lines_score = []
    score_sum = 0
    line_idx = 0
    line = ""
    for i in range(len(word_att_scores)):
        # summerize if meet line separator or the last token
        if ((separator in word_att_scores[i][0]) or (i == (len(word_att_scores) - 1))) and score_sum != 0:
            score_sum += word_att_scores[i][1]
            # append line score as float instead of tensor
            all_lines_score.append(score_sum.item())
            score_sum = 0
            line_idx += 1
        # else accumulate score
        elif separator not in word_att_scores[i][0]:
            line += word_att_scores[i][0]
            score_sum += word_att_scores[i][1]
    return all_lines_score


def clean_special_token_values(all_values, padding=False):
    # special token in the beginning of the seq 
    all_values[0] = 0
    if padding:
        # get the last non-zero value which represents the att score for </s> token
        idx = [index for index, item in enumerate(all_values) if item != 0][-1]
        all_values[idx] = 0
    else:
        # special token in the end of the seq 
        all_values[-1] = 0
    return all_values


def main_cwe(code: list, gpu: bool = False) -> dict:
    """Generate CWE-IDs and CWE Abstract Types Predictions.
    Parameters
    ----------
    code : :obj:`list`
        A list of String functions.
    gpu : bool
        Defines if CUDA inference is enabled
    Returns
    -------
    :obj:`dict`
        A dictionary with four keys, "cwe_id", "cwe_id_prob", "cwe_type", "cwe_type_prob"
        "cwe_id" stores a list of CWE-ID predictions: [CWE-787, CWE-119, ...]
        "cwe_id_prob" stores a list of confidence scores of CWE-ID predictions [0.9, 0.7, ...]
        "cwe_type" stores a list of CWE abstract types predictions: ["Base", "Class", ...]
        "cwe_type_prob" stores a list of confidence scores of CWE abstract types predictions [0.9, 0.7, ...]
    """
    provider = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
    with open("./inference-common/label_map.pkl", "rb") as f:
        cwe_id_map, cwe_type_map = pickle.load(f)
    # load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("./inference-common/tokenizer")
    tokenizer.add_tokens(["<cls_type>"])
    tokenizer.cls_type_token = "<cls_type>"
    model_input = []
    for c in code:
        code_tokens = tokenizer.tokenize(str(c))[:512 - 3]
        source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.cls_type_token] + [tokenizer.sep_token]
        input_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = 512 - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        model_input.append(input_ids)
    device = "cuda" if gpu else "cpu"
    model_input = torch.tensor(model_input, device=device)
    # onnx runtime session
    ort_session = onnxruntime.InferenceSession("./saved_models/onnx_checkpoint/movul.onnx", providers=provider)
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(model_input)}
    cwe_id_prob, cwe_type_prob = ort_session.run(None, ort_inputs)
    # batch_cwe_id_pred (1D list with shape of [batch size]): [pred_1, pred_2, ..., pred_n]
    batch_cwe_id = np.argmax(cwe_id_prob, axis=-1).tolist()
    # map predicted idx back to CWE-ID
    batch_cwe_id_pred = [cwe_id_map[str(idx)] for idx in batch_cwe_id]
    # batch_cwe_id_pred_prob (1D list with shape of [batch_size]): [prob_1, prob_2, ..., prob_n]
    batch_cwe_id_pred_prob = []
    for i in range(len(cwe_id_prob)):
        batch_cwe_id_pred_prob.append(cwe_id_prob[i][batch_cwe_id[i]].item())
    # batch_cwe_type_pred (1D list with shape of [batch size]): [pred_1, pred_2, ..., pred_n]
    batch_cwe_type = np.argmax(cwe_type_prob, axis=-1).tolist()
    # map predicted idx back to CWE-Type
    batch_cwe_type_pred = [cwe_type_map[str(idx)] for idx in batch_cwe_type]
    # batch_cwe_type_pred_prob (1D list with shape of [batch_size]): [prob_1, prob_2, ..., prob_n]
    batch_cwe_type_pred_prob = []
    for i in range(len(cwe_type_prob)):
        batch_cwe_type_pred_prob.append(cwe_type_prob[i][batch_cwe_type[i]].item())
    return {"cwe_id": batch_cwe_id_pred,
            "cwe_id_prob": batch_cwe_id_pred_prob,
            "cwe_type": batch_cwe_type_pred,
            "cwe_type_prob": batch_cwe_type_pred_prob}


def main_sev(code: list, gpu: bool = False) -> dict:
    """Generate CVSS severity score predictions.
    Parameters
    ----------
    code : :obj:`list`
        A list of String functions.
    gpu : bool
        Defines if CUDA inference is enabled
    Returns
    -------
    :obj:`dict`
        A dictionary with two keys, "batch_sev_score", "batch_sev_class"
        "batch_sev_score" stores a list of severity score prediction: [1.0, 5.0, 9.0 ...]
        "batch_sev_class" stores a list of severity class based on predicted severity score ["Medium", "Critical"...]
    """
    provider = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
    # load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("./inference-common/tokenizer")
    model_input = tokenizer(code, truncation=True, max_length=512, padding='max_length',
                            return_tensors="pt").input_ids
    # onnx runtime session
    ort_session = onnxruntime.InferenceSession("./saved_models/onnx_checkpoint/sev_model.onnx", providers=provider)
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(model_input)}
    cvss_score = ort_session.run(None, ort_inputs)
    batch_sev_score = list(cvss_score[0].flatten().tolist())
    batch_sev_class = []
    for i in range(len(batch_sev_score)):
        if batch_sev_score[i] == 0:
            batch_sev_class.append("None")
        elif batch_sev_score[i] < 4:
            batch_sev_class.append("Low")
        elif batch_sev_score[i] < 7:
            batch_sev_class.append("Medium")
        elif batch_sev_score[i] < 9:
            batch_sev_class.append("High")
        else:
            batch_sev_class.append("Critical")
    return {"batch_sev_score": batch_sev_score, "batch_sev_class": batch_sev_class}


def main_repair(code: list, max_repair_length: int = 256, gpu: bool = False) -> dict:
    """Generate vulnerability repair candidates.
    Parameters
    ----------
    code : :obj:`list`
        A list of String functions.
    code : :obj:`int`
        max number of tokens for each repair.
    gpu : bool
        Defines if CUDA inference is enabled
    Returns
    -------
    :obj:`dict`
        A dictionary with one key, "batch_repair"
        "batch_repair" is a list of String, where each String is the repair for one code snippet.
    """
    device = "cuda" if gpu else "cpu"
    # load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("./inference-common/repair_tokenizer")
    tokenizer.add_tokens(["<S2SV_StartBug>", "<S2SV_EndBug>", "<S2SV_blank>", "<S2SV_ModStart>", "<S2SV_ModEnd>"])    
    config = T5Config.from_pretrained("./inference-common/repair_model_config.json")
    model = T5ForConditionalGeneration(config=config)
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load("./saved_models/checkpoint-best-loss/repair_model.bin", map_location=device))
    model.eval()
    input_ids = tokenizer(code, truncation=True, max_length=512, padding='max_length', return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    attention_mask = attention_mask.to(device)
    gen_tokens = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_repair_length)
    batch_repair = tokenizer.batch_decode(gen_tokens)
    for i in range(len(batch_repair)):
        batch_repair[i] = clean_tokens(batch_repair[i])
    return {"batch_repair": batch_repair}


def clean_tokens(tokens):
    tokens = tokens.replace("<pad>", "")
    tokens = tokens.replace("<s>", "")
    tokens = tokens.replace("</s>", "")
    tokens = tokens.strip("\n")
    tokens = tokens.strip()
    return tokens


def to_numpy(tensor):
    """ get np input for onnx runtime model """
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


@app.post('/api/v1/gpu/predict')
def predict_gpu(request: Request):
    functions = asyncio.run(request.json())

    if not functions:
        return {'error': 'No functions to process'}
    else:
        result = json.dumps(main(functions, True))
        return result


@app.post('/api/v1/cpu/predict')
def predict_cpu(request: Request):
    functions = asyncio.run(request.json())

    if not functions:
        return {'error': 'No functions to process'}
    else:
        result = json.dumps(main(functions, False))
        return result


@app.post('/api/v1/gpu/cwe')
def cwe_gpu(request: Request):
    functions = asyncio.run(request.json())

    if not functions:
        return {'error': 'No code to process'}
    else:
        result = json.dumps(main_cwe(functions, True))
        return result


@app.post('/api/v1/cpu/cwe')
def cwe_cpu(request: Request):
    functions = asyncio.run(request.json())

    if not functions:
        return {'error': 'No code to process'}
    else:
        result = json.dumps(main_cwe(functions, False))
        return result


@app.post('/api/v1/gpu/sev')
def sev_gpu(request: Request):
    functions = asyncio.run(request.json())

    if not functions:
        return {'error': 'No code to process'}
    else:
        result = json.dumps(main_sev(functions, True))
        return result


@app.post('/api/v1/cpu/sev')
def sev_cpu(request: Request):
    functions = asyncio.run(request.json())

    if not functions:
        return {'error': 'No code to process'}
    else:
        result = json.dumps(main_sev(functions, False))
        return result


@app.post('/api/v1/gpu/repair')
def repair_gpu(request: Request):
    functions = asyncio.run(request.json())

    if not functions:
        return {'error': 'No code to process'}
    else:
        result = json.dumps(main_repair(functions, 256))
        print(result)
        return result


@app.post('/api/v1/cpu/repair')
def repair_cpu(request: Request):
    functions = asyncio.run(request.json())

    if not functions:
        return {'error': 'No code to process'}
    else:
        result = json.dumps(main_repair(functions, 256))
        print(result)
        return result
