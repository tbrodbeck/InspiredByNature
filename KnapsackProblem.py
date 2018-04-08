class Items:
    def __init__(self):
        self.dict = {}

    def add(self, name, weights, values):
        self.dict[name] = (weights, values)

# create item-list
items = Items()

for i in range(2):
    items.add('stone'+str(i), 40, 20)
items.add('wood', 25, 10)
items.add('fabric', 15, 5)

print(items.dict)
