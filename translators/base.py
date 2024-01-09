import abc

class BaseTranslator(abc.ABC):
    def __init__(self, device, quant4, quant4_config, quant8, max_length):
        self.device = device
        self.quant4 = quant4
        self.quant4_config = quant4_config
        self.quant8 = quant8
        self.max_length = max_length

    @abc.abstractmethod
    def translate(self, texts, source_lang, target_lang):
        pass