from abc import abstractmethod

class Model:

  @abstractmethod
  def detect(self, orig_img): pass