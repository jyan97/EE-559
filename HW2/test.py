with open('wine_train.csv', 'r') as train_data:
    reader = train_data.read().splitlines()
    x_data, y_data, label = [], [], []
    for rows in reader:
        rows = rows.split(',')
        x_data.append(rows[0])
        y_data.append(rows[1])
        label.append(rows[13])
    x_data = list(map(float, x_data))
    y_data = list(map(float, y_data))
    label = list(map(int, label))

    data_x = [x_data[m] for m in range(len(label)) if label[m] == 1]
    data_y = [m for m in y_data if label[y_data.index(m)] == 1]

    complement_x = [m for m in x_data if label[x_data.index(m)] != 1]
    complement_y = [m for m in y_data if label[y_data.index(m)] != 1]

# print(sum(data_x)/len(data_x))
print(data_x)