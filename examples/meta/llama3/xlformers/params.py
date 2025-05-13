import copy


class DeepCopyDict(dict):
    def __getitem__(self, item):
        res = super(DeepCopyDict, self).__getitem__(item)
        if callable(res):
            return res()
        return copy.deepcopy(res)

    def __setitem__(self, key, value):
        if key in self:
            raise ValueError(f"{key} already in ConfStore")

        super(DeepCopyDict, self).__setitem__(key, value)


ConfStore = DeepCopyDict()
