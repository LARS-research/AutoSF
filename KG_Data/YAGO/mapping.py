trainList = []
validList = []
testList = []
entities = {}
with open('train', 'r') as f:
    tmp = f.readline()
    total = int(tmp.strip())
    for i in range(total):
        tmp = f.readline()
        h, r, t = tmp.strip().split()
        #h, t, r = int(h), int(t), int(r)
        if h not in entities:
            entities[h] = 1
        if t not in entities:
            entities[t] = 1
        trainList.append((h,t,r))

with open('valid', 'r') as f:
    tmp = f.readline()
    total = int(tmp.strip())
    for i in range(total):
        tmp = f.readline()
        h, r, t = tmp.strip().split()
        #h, t, r = int(h), int(t), int(r)
        if h not in entities:
            entities[h] = 1
        if t not in entities:
            entities[t] = 1
        validList.append((h,t,r))

with open('test', 'r') as f:
    tmp = f.readline()
    total = int(tmp.strip())
    for i in range(total):
        tmp = f.readline()
        h, r, t = tmp.strip().split()
        #h, t, r = int(h), int(t), int(r)
        if h not in entities:
            entities[h] = 1
        if t not in entities:
            entities[t] = 1
        testList.append((h,t,r))

print(len(trainList))
entity = entities.keys()
print(len(entity), min(entity), max(entity))

with open('train2id.txt', 'w+') as f:
    total = len(trainList)
    f.write(str(total) + '\n')
    for i in range(total):
        trip = trainList[i]
        line = trip[0] + '\t' + trip[1] + '\t' + trip[2] + '\n'
        f.write(line)

with open('valid2id.txt', 'w+') as f:
    total = len(validList)
    f.write(str(total) + '\n')
    for i in range(total):
        trip = validList[i]
        line = trip[0] + '\t' + trip[1] + '\t' + trip[2] + '\n'
        f.write(line)


with open('test2id.txt', 'w+') as f:
    total = len(testList)
    f.write(str(total) + '\n')
    for i in range(total):
        trip = testList[i]
        line = trip[0] + '\t' + trip[1] + '\t' + trip[2] + '\n'
        f.write(line)

