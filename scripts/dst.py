from itertools import chain, combinations
import numpy as np

from mdet.msg import GridDST, GridDSTMetaData, GridDSTMass

def powerset(s):
    powerset = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    return frozenset([frozenset(x) for x in powerset])


def make_set(val):
    if type(val) in (int, str, float, bool):
        return frozenset([val])
    else:
        return frozenset(val)

class MassFunction(dict):

    def __init__(self, universe=None, instance=None):
        super(MassFunction, self).__init__()
        if universe is None:
            universe = instance.get_universe()
        self._universe = make_set(universe)
        self._powerset = powerset(universe)

    def get_universe(self):
        return self._universe

    def get_powerset(self):
        return self._powerset

    def normalize_universe(self):
        sum = 0.0
        for item in self._powerset - frozenset([self._universe]):
            sum += self[item]

        self[self._universe] = np.maximum(0.0, 1.0 - sum)  # clip negative to zero


    def __setitem__(self, key, value):
        key = make_set(key)
        super(MassFunction, self).__setitem__(key, value)

    def __contains__(self, item):
        item = make_set(item)
        return item in self._superset

    def __getitem__(self, item):
        item = make_set(item)
        if super(MassFunction, self).__contains__(item):
            return super(MassFunction, self).__getitem__(item)
        else:
            return 0.0

    def is_focal(self, key):
        val = self[key]
        return type(val) is not float or val != 0.0

    def get_focal_sets(self, include_universe=True):
        return frozenset([item for item in self.get_powerset() if self.is_focal(item) and (include_universe or item != self._universe)])

    def belief(self, key):
        key = make_set(key)
        result = 0.0
        for item in self.get_powerset():
            if key.issubset(item):
                result += self[item]

        return result

    def plausibility(self, key):
        key = make_set(key)

        result = 0.0
        for item in self.get_powerset():
            intersection = key.intersection(item)
            if len(intersection) != 0:
                result += self[item]
        return result

    def conj_dempster(self, m):
        if type(m) is not type(m):
            raise ValueError("Only MassFunction allowed for combination")
        if m.get_powerset() != self.get_powerset():
            raise ValueError("Power sets need to be equal for combination")

        conflict = 0.0

        for s_item in self.get_powerset():
            for m_item in self.get_powerset():
                intersection = s_item.intersection(m_item)
                if intersection == frozenset():
                    conflict += self[s_item] * m[m_item]

        result = self.__class__(instance=self)

        for s_item in self.get_powerset():
            for m_item in self.get_powerset():
                intersection = s_item.intersection(m_item)
                if intersection != frozenset():
                    if self.is_focal(s_item) and m.is_focal(m_item):  # don't keep arrays of 0.0
                        result[intersection] += self[s_item] * m[m_item]

        return result, conflict

    def josang_cumulative(self, m):

        # update self

        m.normalize_universe()
        self.normalize_universe()

        self['D'] = m['D']
        m['D'] = 0.0

        uni = self.get_universe()
        items = uni - frozenset(['D', ''])

        sum = 0.0

        for item in items:

            self[item] = np.where(np.logical_or(m[uni] > 1e-7, self[uni] > 1e-7),
                                  (self[item] * m[uni] + self[uni] * m[item]) / (self[uni] + m[uni] - self[uni] * m[uni]),
                                  0.5*self[item] + 0.5*m[item]
                                  )
            sum += self[item]

        self[uni] = np.where(np.logical_or(m[uni] > 1e-7, self[uni] > 1e-7),
                              (self[uni] * m[uni]) / (self[uni] + m[uni] - self[uni] * m[uni]),
                              0.0
                              )

        sum += self[uni]

        norm_w = np.where(sum > 0.0, (1.0 - self['D']) / sum, 1.0)

        for item in items:
            self[item] *= norm_w

        self[uni] *= norm_w

        self.normalize_universe()



class GridMassFunction(MassFunction):

    def __init__(self, universe=None, metadata=None, instance=None):
        super(self.__class__, self).__init__(universe, instance=instance)
        if metadata is not None:
            self._grid_metadata = metadata
        else:
            self._grid_metadata = instance.get_metadata()

    def get_metadata(self):
        return self._grid_metadata

    @classmethod
    def fromMessage(cls, msg):

        universe = msg.universe
        metadata = msg.metadata
        m = cls(universe, metadata)
        for mass in msg.masses:
            data = np.array(mass.data, dtype=np.float32).reshape((metadata.width, metadata.height))
            m[mass.item] = data

        return m

    def asMessage(self, header):
        metadata = GridDSTMetaData()
        metadata.resolution = self._grid_metadata.resolution
        metadata.width = self._grid_metadata.width
        metadata.height = self._grid_metadata.height
        metadata.origin = self._grid_metadata.origin

        msg = GridDST()
        msg.metadata = metadata
        msg.header = header
        msg.universe = [str(item) for item in self._universe]

        for item in self.get_focal_sets():
            mass = GridDSTMass()
            mass.item = [str(x) for x in item]
            mass.data = self[item].ravel().tolist()
            msg.masses.append(mass)

        return msg