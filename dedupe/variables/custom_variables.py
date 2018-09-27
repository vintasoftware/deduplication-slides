import logging

from dedupe.variables.string import ShortStringType
import jellyfish


logger = logging.getLogger(__name__)


class JaroWinklerType(ShortStringType):
    type = "JaroWinkler"

    def __init__(self, definition):
        super().__init__(definition)

        self.comparator = jellyfish.jaro_winkler
