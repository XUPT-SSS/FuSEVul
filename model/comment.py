import pandas as pd
import openai

openai.api_key = 'api-key'
openai.api_base = "https://api.chatanywhere.com.cn/v1"

data = pd.read_csv('small_trvd/small_our_val.csv')
#print(data)
data_1 = data['text'].values.tolist()
data_2 = data['label'].values.tolist()
data2 = pd.read_csv('small_trvd/ss_val.csv')
data_3 = data2['text'].values.tolist()
data_4 = data2['label'].values.tolist()
k = len(data_3)
#data_3 = []
#data_4 = []
#k = 0
#print(data_1[1])gpt-3.5-turbo-0125 gpt-4-0125-preview

q = 'Please explain what the following C/C++ function does and pay attention to any potential security risks in the code:\n'
dict = {}
while k < len(data_1):
    rsp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "system", "content": "A programmer with 10 years of coding experience"},
                  {"role": "user", "content": q + data_1[k]}]
    )
    msg = rsp.get("choices")[0]["message"]["content"]
    p = msg.replace('\n', '')
    data_3.append(p)
    data_4.append(data_2[k])
    dict['text'] = data_3
    dict['label'] = data_4
    df = pd.DataFrame(dict)
    df.to_csv('small_trvd/ss_val.csv', index=None)
    k += 1
    print(k)
