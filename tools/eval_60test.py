import pandas as pd
import re

df = pd.read_csv('result.csv', encoding='gbk')
# print(df['file'][0])
total = 0
fail = 0
success = 0
class_total = {}
class_fail = {}
class_success = {}
for i in range(60):
    class_total[str(i+1)] = 0
    class_fail[str(i+1)] = 0
    class_success[str(i+1)] = 0

for i in range(len(df)):
    number = re.findall("\d+", str(df['file'][i]))
    total += 1
    class_total[number[0]] += 1
    if number[0] != str(df['class'][i]):
        fail += 1
        class_fail[number[0]] += 1
    else:
        success += 1
        class_success[number[0]] += 1

for i in range(60):
    # print("{} {}/{}, {:.2f}".format(i+1, class_fail[str(i+1)], class_total[str(i+1)], class_fail[str(i+1)]/float(class_total[str(i+1)])))
    print("{} {}/{}, {:.2f}".format(i+1, class_success[str(i+1)], class_total[str(i+1)], class_success[str(i+1)]/float(class_total[str(i+1)])))

# print("{}/{}, {:.2f}".format(fail, total, fail/float(total)))
print("{}/{}, {:.2f}".format(success, total, success/float(total)))
