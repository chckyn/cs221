class Classifier(object):
    def __init__(self, train_patterns, binary_train_patterns, \
            hidden_patterns, ae_patterns):

        self.train_patterns = train_patterns
        self.binary_train_patterns = binary_train_patterns
        self.hidden_patterns = hidden_patterns
        self.ae_patterns = ae_patterns
        

    def __repr__(self):
        return "Classifier(%r, %r, %r, %r)" % \
                (self.train_patterns,
                self.binary_train_patterns,
                self.hidden_patterns,
                self.ae_patterns,)
