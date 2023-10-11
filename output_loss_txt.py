import csv

readlines = open('./data/vae_loss_100.txt', 'r').readlines()
for i in readlines:
    count = 0
    index = 0
    for j in i:
        if j == ':':
            count += 1
        if count == 4:
            # print(i[index+1 : -1])
            data = float(i[index+1 : -1])
            break
        index += 1
    print(data)
    newRow = [data]
    csvFile = open("./vae_loss_list.csv", 'a', newline='', encoding='utf-8')
    writer = csv.writer(csvFile)
    writer.writerow(newRow)  # 数据写入文件中zz
