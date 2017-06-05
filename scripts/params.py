
class Params(object):

    _params = [] # list of tuple(name, default_value)

    def _copy_values(self):
        self._values = {}
        for param in self._params:
            param_name = param[0] # allow variable number of items in tuples
            param_default_val = param[1]
            self._values[param_name] = param_default_val

    def __init__(self):
        self._copy_values()

    def __getattr__(self, item):
        if item.startswith('_'):
            return super(Params, self).__getattribute__(item)
        else:
            return self._values[item]

    def __setattr__(self, key, value):
        if hasattr(self, key) or key.startswith('_'):
            super(Params, self).__setattr__(key, value)
        else:
            if key in [param[0] for param in self._params]:
                self._values[key] = value
            else:
                raise ValueError("Cannot set param, param {0} not found".format(key))


class RosParams(Params):

    # _params should tuple of (name, default_value, ros_name)

    def _load_ros_params(self, namespace):
        for param_name, default_value, type, ros_name in self._params:
            setattr(self, param_name, type(self._rospy.get_param(namespace + ros_name, default_value)))

    def __init__(self, rospy=None, namespace='~'):
        Params.__init__(self)
        if rospy is not None:
            self._rospy = rospy
            self._load_ros_params(namespace)


