import requests
import re

def parse_kb(input,):

    regex = r"(?<=c/en/)\w*"

    obj = requests.get('http://api.conceptnet.io/related/c/en/'+input+'?filter=/c/en')

    test_string = obj.text
    match = re.findall(regex,test_string)
    lisst = []
    for i in range(len(match)):
        a = match[i].split('_')
        lisst.append(a)

    flat_list = []
    for sublist in lisst:
        for item in sublist:
            if item != str(input) and item != "and" and item != "of":
                flat_list.append(item)

    if flat_list:
        return(flat_list)
    else:
        print("Unsuccess")

