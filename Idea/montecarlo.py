import random as r

p = 0.49

done = False
iteration = 0
qs = []

while (not done):

    iteration += 1

    q = (1-p)*r.random() + p

    print(q)

    if (iteration == 100):
        done = True

    qs.append(q)
    print(q)

q = min(qs)
print(q)
if(.5 < q):
    print(True)
else:
    print(False)
