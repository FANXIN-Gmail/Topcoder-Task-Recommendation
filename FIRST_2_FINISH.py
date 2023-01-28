import json
import requests
import csv
import time


Dict = {
    "id": "",
    "name": "",
    "technologies": "",
    "platforms": "",
    "numRegistrants": "",
    "numSubmitters": "",
    "duration": "",
    "prizes": "",
    "winners": ""
}

# offset = 0
Offset = 0
Count = 0

with open("FIRST_2_FINISH.csv", "w", newline="", encoding="utf-8") as FIRST_2_FINISH_csv:

    header = list(Dict.keys())
    dict_writer = csv.DictWriter(FIRST_2_FINISH_csv, header)
    dict_writer.writeheader()

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36'}

    while True:

        offset = str(Offset * 50)
        Offset += 1

        html = requests.get('https://api.topcoder.com/v4/challenges/?filter=status%3DCOMPLETED&limit=50&offset=' + offset, headers=headers)

        if html.status_code == 200:

            html.encoding = 'utf-8'
            data = json.loads(html.text)
            # print(offset, Count, data['result']['metadata']['totalCount'])

            if data['result']['metadata']['totalCount'] == 0:
                break
            else:
                for i in range(data['result']['metadata']['totalCount']):
                    if data['result']["content"][i]["subTrack"] == "FIRST_2_FINISH":

                        Dict["id"] = data['result']['content'][i]['id']
                        Dict["name"] = data['result']['content'][i]['name']

                        try:
                            Dict["technologies"] = data['result']['content'][i]['technologies']
                        except KeyError as error:
                            Dict["technologies"] = "None"
                            print("KeyError: ", error)
                            continue

                        try:
                            Dict["platforms"] = data['result']['content'][i]["platforms"]
                        except KeyError as error:
                            Dict["platforms"] = "None"
                            print("KeyError: ", error)
                            continue

                        try:
                            Dict["numRegistrants"] = data['result']['content'][i]["numRegistrants"]
                        except KeyError as error:
                            Dict["numRegistrants"] = "None"
                            print("KeyError: ", error)
                            continue

                        try:
                            Dict["numSubmitters"] = data['result']['content'][i]["numSubmitters"]
                        except KeyError as error:
                            Dict["platforms"] = "None"
                            print("KeyError: ", error)
                            continue

                        for j in data['result']['content'][i]["allPhases"]:
                            if j["phaseType"] == "Submission":
                                Dict["duration"] = j["duration"]

                        try:
                            Dict["prizes"] = data['result']['content'][i]["prizes"]
                        except KeyError as error:
                            Dict["prizes"] = "None"
                            print("KeyError: ", error)
                            continue

                        Dict["winners"] = []

                        try:
                            for k in data['result']['content'][i]["winners"]:
                                Dict["winners"].append(k["handle"])
                        except KeyError as error:
                            Dict["winners"] = "None"
                            print("KeyError: ", error)
                            continue

                        with open("FIRST_2_FINISH.txt", "a", newline="", encoding="utf-8") as FIRST_2_FINISH_txt:
                            FIRST_2_FINISH_txt.write("https://www.topcoder.com/challenges/" + str(Dict["id"]) + "\n")
                            FIRST_2_FINISH_txt.flush()

                        dict_writer.writerow(Dict)
                        FIRST_2_FINISH_csv.flush()
                        Count += 1
                    else:
                        continue

                print(offset, Count, data['result']['metadata']['totalCount'])
                time.sleep(1)
        else:
            print(offset, html.status_code)
            break
