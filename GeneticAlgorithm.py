class Genetic_Algotihm:

    def __init__(self, initializer, selector, recombiner, mutator, replacer):
        self.inializer = initializer
        self.selector = selector
        self.recombiner = recombiner
        self.mutator = mutator
        self.replacer = replacer

