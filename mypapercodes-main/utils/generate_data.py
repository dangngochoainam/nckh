from preprocess.preprocess import execute_new_data

def generate_data(data, targets):
    new_data = []
    new_target = []
    i = 1
    for d in data:
        new_data.append(execute_new_data(d))
        print(i)
        i = i + 1

    for t in targets:
        new_target.append(t)

    data = {'feature': new_data, 'target': new_target}

    return data