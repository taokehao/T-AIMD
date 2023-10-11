from Transformer.myfunction import draw_loss

loss_list = []
readlines = open('../data/vae_loss.txt', 'r').readlines()
for i in readlines:
    index = 0
    cout = 0
    for j in i:
        if j == ':':
            cout += 1
        if cout == 4:
            print(i[index+1:-1])
            loss_list.append(float(i[index+1:-1]))
            break
        index += 1

draw_loss(loss_list,1000)
