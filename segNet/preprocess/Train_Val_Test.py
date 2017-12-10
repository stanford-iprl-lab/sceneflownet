import numpy as np
np.random.seed(42)

class Train_Val_Test(object):
  def __init__(self, total_lists, splitting=[0.7, 0.1, 0.2]):
    self._num_examples = len(total_lists)
    tmp_permutations = np.random.permutation(self._num_examples)
    self.num_train = int(splitting[0]*self._num_examples)
    self.num_val = int(splitting[1]*self._num_examples)
    self.num_test = self._num_examples - self.num_train - self.num_val
    self._train = total_lists[tmp_permutations[0:self.num_train]]
    self._val   = total_lists[tmp_permutations[self.num_train: self.num_train+self.num_val]]
    self._test  = total_lists[tmp_permutations[self.num_train+self.num_val:]]
