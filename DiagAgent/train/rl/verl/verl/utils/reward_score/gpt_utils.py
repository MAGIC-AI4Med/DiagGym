import requests
import json
import csv
import ast
import hashlib
import html
from Crypto.Cipher import AES
from binascii import b2a_hex, a2b_hex

# doc:  https://yuque.antfin-inc.com/sjsn/biz_interface/pdxkcxwic4kdarrf

def aes_encrypt(data, key):
    """aes加密函数，如果data不是16的倍数【加密文本data必须为16的倍数！】，那就补足为16的倍数
    :param key:
    :param data:
    """
    iv = "1234567890123456"
    cipher = AES.new(key.encode('utf-8'), AES.MODE_CBC, iv.encode('utf-8'))  # 设置AES加密模式 此处设置为CBC模式
    block_size = AES.block_size

    # 判断data是不是16的倍数，如果不是用b'\0'补足
    if len(data) % block_size != 0:
        add = block_size - (len(data) % block_size)
    else:
        add = 0
    data = data.encode('utf-8') + b'\0' * add
    encrypted = cipher.encrypt(data)  # aes加密
    result = b2a_hex(encrypted)  # b2a_hex encode  将二进制转换成16进制
    return result.decode('utf-8')


def aes_decode(data, key):
    """aes解密
    :param key:
    :param data:
    """
    iv = '1234567890123456'
    cipher = AES.new(key.encode('utf-8'), AES.MODE_CBC, iv.encode('utf-8'))
    result2 = a2b_hex(data)  # 十六进制还原成二进制
    decrypted = cipher.decrypt(result2)
    return decrypted.rstrip(b'\0')  # 解密完成后将加密时添加的多余字符'\0'删除


def parse_env_file(file_path):
    """模仿 docenv 库，加载并解析一个配置文件
    """
    env_vars = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip()
    return env_vars

# Load environment variables from .env
import os
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
env_vars = parse_env_file(os.path.join(current_dir, 'credentials.env'))

# Access the values using the keys
GPT35_VISIT_DOMAIN = env_vars["GPT35_VISIT_DOMAIN"]
GPT35_VISIT_BIZ = env_vars["GPT35_VISIT_BIZ"]
GPT35_VISIT_BIZLINE = env_vars["GPT35_VISIT_BIZLINE"]

GPT4_VISIT_DOMAIN = env_vars["GPT4_VISIT_DOMAIN"]
GPT4_VISIT_BIZ = env_vars["GPT4_VISIT_BIZ"]
GPT4_VISIT_BIZLINE = env_vars["GPT4_VISIT_BIZLINE"]

API_KEY = env_vars["API_KEY"]
AES_KEY = env_vars["AES_KEY"]
USER_NAME = env_vars["USER_NAME"]

assert USER_NAME != "", "Adding USER_NAME in the credentials.env file is required."
 
def get_default_config(model, temperature=0.8, top_p=1, frequency_penalty=1.1, presence_penalty=1.1):
    """ return the params needed for making a request to chatgpt api

        *************
        *** DON'T *** Don't change the default params here
        *************
        
        Make changes outside, e.g.     
            param = get_default_config()
            param["queryConditions"]["model"] = "gpt-3.5-turbo(0301)"
            param["queryConditions"]["messages"][0]["content"] = "explain what is chatgpt"            
    """
    if model in  ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-instruct"]:
        VISIT_DOMAIN = GPT35_VISIT_DOMAIN
        VISIT_BIZ = GPT35_VISIT_BIZ
        VISIT_BIZLINE = GPT35_VISIT_BIZLINE
    elif model in ["gpt-4", "gpt-4-32k", "gpt-4-turbo"]:
        VISIT_DOMAIN = GPT4_VISIT_DOMAIN
        VISIT_BIZ = GPT4_VISIT_BIZ
        VISIT_BIZLINE = GPT4_VISIT_BIZLINE
    else:
        assert False, "Unknown model"

    param = {
        "serviceName": "chatgpt_prompts_completions_query_dataview",
        "visitDomain": VISIT_DOMAIN,
        "visitBiz": VISIT_BIZ,
        "visitBizLine": VISIT_BIZLINE,
        "cacheInterval": -1, # 不缓存
        "queryConditions": {
            # "url": "%s",
            "requestName": USER_NAME,
            "model": model,   # gpt-3.5-turbo(0301), gpt-3.5-turbo-0301, gpt-3.5-turbo-16k(0613) gpt-4
            "api_key": API_KEY,
            "messages": [{
                "role": "user",
                "content": None # placeholder for prompt
            }],
            "scene_code": "cto_userunderstand",
            "temperature": "0.8" # 介于 0 和 2 之间。较高的值（如 0.8）将使输出更加随机，而较低的值（如 0.2）将使输出更加集中和确定。
            # "max_tokens": # 1、gpt-3.5-turbo模型，支持4,096 token 2、gpt-3.5-turbo-16k模型，支持16,384 token
        }
    }
    return param

def cal_msg_key(msg):
    return hashlib.sha256(str(msg+"123").encode('utf-8')).hexdigest()

def ask_chatgpt_async_send(param):
    param["serviceName"] = "asyn_chatgpt_prompts_completions_query_dataview"
    param["queryConditions"]["outputType"] = "PULL"
    param["queryConditions"]["n"] = 1
    param["queryConditions"]["messageKey"] = cal_msg_key(param["queryConditions"]["messages"][0]["content"] + param["queryConditions"]["temperature"])
    return ask_chatgpt(param, False)

def ask_chatgpt_async_fetch(param):
    param["serviceName"] = "chatgpt_response_query_dataview"
    param["queryConditions"] = {
        "messageKey": cal_msg_key(param["queryConditions"]["messages"][0]["content"] + param["queryConditions"]["temperature"])
    }
    return ask_chatgpt(param, if_parse_results=True, is_async_format=True)




def ask_chatgpt(param, if_parse_results=True, is_async_format=False):
    # wrapper for the chatgpt api:  https://yuque.antfin-inc.com/sjsn/biz_interface/pdxkcxwic4kdarrf
    url = 'https://zdfmng.alipay.com/commonQuery/queryData'
    data = json.dumps(param) #% url.encode('utf8')
    encrypted_data = aes_encrypt(data, AES_KEY) # 密钥
    post_data = {
        "encryptedParam": encrypted_data
    }
    headers = {
        'Content-Type': 'application/json'
    }
    
    response = requests.post(url, data=json.dumps(post_data), headers=headers)

    if not response.json()["data"]["success"]:
        result = response.json()
        result["data"]["errorMessage"] = html.unescape(result["data"]["errorMessage"])
        assert False, f"Response failed! with msg: {result}"

    # 如果异步的发送请求，不进行解析
    if not if_parse_results:
        # print(f"send request {response.json()}")
        return None
    else:
        # print(f"fetch request {response.json()}")
        if is_async_format:
            ast_str = ast.literal_eval("'" + response.json()["data"]["values"]["response"] + "'")
        else:
            ast_str = ast.literal_eval("'" + response.json()["data"]["values"]["data"] + "'")
        js = html.unescape(ast_str)
        # print(f"fetch request 2 {js}")
        data = json.loads(js)

        content = data["choices"][0]["message"]["content"]
        prompt_tokens = data["usage"]["prompt_tokens"]
        total_tokens = data["usage"]["total_tokens"]
        return content

import pandas as pd
def read_prompts(csv_file_path):
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)
    # 将第一列转换为列表
    return df.iloc[:, 0].tolist()

    # prompts = []
    # with open(csv_file_path, 'r') as file:
    #     # Create a CSV reader
    #     reader = csv.reader(file)

    #     # Iterate over each row in the CSV file
    #     for row in reader:
    #         prompts.append(row[0])
    # return prompts[1:]

# @Deprecated, to be replaced by the ask_chatgpt function
def get_asyn_config(model):
    param = get_default_config(model)
    param["serviceName"] = "asyn_chatgpt_prompts_completions_query_dataview"
    param["queryConditions"]["outputType"] = "PULL"
    param["queryConditions"]["n"] = 1
    param["queryConditions"]["messageKey"] = None # placeholder for message key
    return param

# @Deprecated, to be replaced by the ask_chatgpt function
def get_fetch_config(model):
    param = get_default_config(model)
    param["serviceName"] = "chatgpt_response_query_dataview"
    param["queryConditions"] = {
        "messageKey": None # placeholder for message key
    }
    return param

# @Deprecated, to be replaced by the ask_chatgpt function
def send_request(param):
    url = 'https://zdfmng.alipay.com/commonQuery/queryData'
    data = json.dumps(param)
    post_data = {
        "encryptedParam": aes_encrypt(data, AES_KEY)
    }
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(url, data=json.dumps(post_data), headers=headers)
    return response

# @Deprecated, to be replaced by the ask_chatgpt function
def parse_response(response):
    try:
        response_js = response.json()
    except JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"response is {response.text}")
        raise Exception(f"调用结果不是json")
    if not response_js['success']:
        raise Exception(f"调用失败, 结果为: {response_js}")
    return response_js

# @Deprecated, to be replaced by the ask_chatgpt function
def parse_fetch_result(response):
    response_js = parse_response(response)
    x = response_js["data"]["values"]["response"]
    ast_str = ast.literal_eval("'" + x + "'")
    js = html.unescape(ast_str)
    data = json.loads(js)
    content = data['choices'][0]['message']['content']
    return content
