num_layers = 12
temp = []
a = [0] * num_layers

for i in range(num_layers):
    b = [0] * num_layers
    b[i] = 1
    temp.append(",".join([str(x) for x in b]))

print(temp)