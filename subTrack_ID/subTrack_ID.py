import json
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36'}
#
# with open('subTrack.txt', 'r') as subTrack_file:
#     subTrack = subTrack_file.readlines()
#     for i in range(len(subTrack)):
#         subTrack[i] = subTrack[i][:-1]
#         subTrack[i] += ".txt"
#
# Dict = {}
#
# for i in subTrack:
#     Dict[i[:-4]] = open(i, 'w')
#
# offset = 0
# Offset = 0
#
# while True:
#     offset = str(Offset*50)
#     Offset += 1
#     html = requests.get('https://api.topcoder.com/v4/challenges/?filter=status%3DCOMPLETED&limit=50&offset=' + offset, headers=headers)
#
#     if html.status_code == 200:
#         html.encoding = 'utf-8'
#         data = json.loads(html.text)
#         print(offset, data['result']['metadata']['totalCount'])
#         if data['result']['metadata']['totalCount'] == 0:
#             break
#         else:
#             for i in range(data['result']['metadata']['totalCount']):
#                 try:
#                     Dict[data['result']['content'][i]['subTrack']].write(str(data['result']['content'][i]['id']) + "\n")
#                 except KeyError as error:
#                     print("error: ", error)
#                     with open('Extra_subTrack.txt', 'a') as Extra_subTrack_file:
#                         Extra_subTrack_file.write(data['result']['content'][i]['subTrack'] + "\n")
#                         Extra_subTrack_file.flush()
#                 else:
#                     Dict[data['result']['content'][i]['subTrack']].flush()
#
#             time.sleep(1)
#
#     else:
#         print(offset, html.status_code)
#         break
#
# for i in Dict.values():
#     i.close()


with open('subTrack.txt', 'r') as subTrack_file:
    subTrack = subTrack_file.readlines()
    for i in range(len(subTrack)):
        subTrack[i] = subTrack[i][:-1]
        subTrack[i] += ".txt"

    Dict = {}

    for i in subTrack:
        Dict[i[:-4]] = open(i, 'r')

    for key, value in Dict.items():
        content = value.readlines()
        Dict[key] = len(content)
        value.close()

    Count = pd.Series(Dict)
    Count.plot(kind='bar')
    x = np.arange(30)
    y = np.array(list(Dict.values()))
    for a, b in zip(x, y):
        plt.text(a, b+0.05, '%.0f' % b, ha='center', va='bottom', fontsize=5)
    plt.show()

    All = 0

    for value in Dict.values():
        All += value

    print(All)


